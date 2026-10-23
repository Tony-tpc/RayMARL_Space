"""Fixed-policy, fixed-MC-data UC value experiment, independent of PPO training.

Run with the repository Python. The preset is recorded before sampling: three
architectures, three optimization seeds, twenty epochs, identical loss chunks.
Only the exported joint actors generate trajectories. No fitted probe is ever
used by an actor, environment, return label, or candidate-selection routine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "RLAlgo/src"), str(ROOT / "environments/sn_energy_market/src")]

from raymarl.algorithms.stackelberg.critic_probe import CriticProbe
from raymarl.algorithms.stackelberg.inference import ActorBundle
from raymarl.algorithms.stackelberg.rl_module import ValueNetwork

from experiments.sn_energy_market.evaluation import (
    VALIDATION_SEEDS,
    error_statistics,
    stratified_statistics,
    terminal_returns,
)
from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)

SOURCE = ROOT / "outputs/ausgrid-layernorm-pilot/20261020"
DEFAULT_OUTPUT = ROOT / "outputs/ausgrid-frozen-critic/20261021"
KINDS, FIT_SEEDS = ("mlp", "time", "gru"), (11, 29, 47)
EPOCHS, CHUNK_LENGTH, CHUNKS_PER_BATCH = 20, 32, 8
SCALE, LR, GRAD_CLIP, LOSS_COEFFICIENT = 20.0, 1e-3, 0.5, 0.5
CALENDAR_FIELDS = (
    "month_fraction_elapsed", "remaining_steps_div_1488", "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = list((ROOT / "environments/sn_energy_market/src").rglob("*.py"))
    files += list((ROOT / "RLAlgo/src/raymarl/algorithms/stackelberg").glob("*.py"))
    files += [Path(__file__), ROOT / "experiments/sn_energy_market/profile_dataset.py",
              ROOT / "experiments/sn_energy_market/training_env.py",
              ROOT / "experiments/sn_energy_market/evaluation.py"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(files))}


def actor_digest(bundle):
    digest = hashlib.sha256()
    for agent, module in sorted(bundle.modules.items()):
        for name, value in sorted(module.actor.state_dict().items()):
            digest.update(f"{agent}/{name}/{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def public_calendar(timestamp, period, total):
    """Only a known calendar and public episode horizon, before the UC action."""
    hour = timestamp.hour + timestamp.minute / 60
    phases = (hour / 24, timestamp.weekday() / 7, (timestamp.month - 1) / 12)
    return np.asarray(
        [period / total, (total - period) / 1488,
         *[f(2 * np.pi * phase) for phase in phases for f in (np.sin, np.cos)]],
        dtype=np.float32,
    )


def load_context():
    contract = json.loads((SOURCE / "checkpoint/learning_contract.json").read_text(encoding="utf-8"))
    dataset = ProfileDataset(ROOT / "datasets/ausgrid/processed/manifest.json", splits=("train", "validation"))
    factory = DatasetEnvFactory(dataset, seed=42)
    if factory.get_contract() != contract["scenario"]:
        raise ValueError("Current scenario differs from the frozen checkpoint")
    bundle = ActorBundle(SOURCE / "actors")
    for agent, module in bundle.modules.items():
        module.requires_grad_(False)
        path = SOURCE / f"checkpoint/learner_group/learner/rl_module/{agent}/module_state.pkl"
        state = pickle.loads(path.read_bytes())
        for key, value in module.actor.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), state[f"actor.{key}"])
    uc_state = pickle.loads((SOURCE / "checkpoint/learner_group/learner/rl_module/uc/module_state.pkl").read_bytes())
    observation_dim = bundle.modules["uc"].observation_space.shape[0]
    value = ValueNetwork(observation_dim, contract["model"]["hidden_dim"])
    value.load_state_dict({key.removeprefix("value."): torch.as_tensor(item)
                          for key, item in uc_state.items() if key.startswith("value.")})
    value.eval().requires_grad_(False)
    return contract, factory, bundle, value


@torch.no_grad()
def collect_episode(factory, bundle, old_value, seed, gamma, *, progress=None):
    env = factory(seed)
    obs = env.reset(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    fields = {key: [] for key in (
        "obs", "calendar", "uc_actions", "consumer_actions", "rewards", "all_rewards",
        "terminated", "periods", "timestamps", "checkpoint_predictions",
    )}
    totals = {}
    start = time.perf_counter()
    while True:
        period, timestamp = env.engine.state.period_id, env.engine.state.timestamp
        fields["obs"].append(obs.copy())
        fields["calendar"].append(public_calendar(timestamp, period, env.config.system.horizon_steps))
        fields["timestamps"].append(timestamp.isoformat())
        fields["periods"].append(period)
        fields["checkpoint_predictions"].append(float(old_value(torch.as_tensor(obs)[None])[0] * SCALE))
        decision = env.publish(bundle.uc_controls(obs, env.uc_bounds(), deterministic=False))
        actions = {key: bundle.consumer_action(
            key, decision.consumer_observations[key], decision.consumer_bounds[key], deterministic=False,
        ) for key in env.consumer_ids}
        result = env.step(actions)
        fields["uc_actions"].append(decision.uc_action)
        fields["consumer_actions"].append(np.stack([actions[key] for key in env.consumer_ids]))
        fields["rewards"].append(result.rewards["uc"])
        fields["all_rewards"].append([result.rewards[key] for key in ("uc", *env.consumer_ids)])
        fields["terminated"].append(result.terminated)
        for key, val in result.metrics.items():
            totals[key] = totals.get(key, 0.0) + val
        if progress and (period + 1) % 384 == 0:
            progress(seed, period + 1)
        if result.terminated:
            break
        if result.data_exhausted:
            raise AssertionError("A complete finite month ended without a true terminal")
        obs = result.next_uc_observation
    arrays = {key: np.asarray(val) for key, val in fields.items()}
    arrays["targets"] = terminal_returns(arrays["rewards"], arrays["terminated"], gamma)
    validate_episode(arrays, gamma)
    assert len(arrays["rewards"]) == env.config.system.horizon_steps
    metadata = {
        "seed": seed, "month": timestamp.strftime("%Y-%m"),
        "split": factory._by_seed[seed].split, "periods": len(arrays["rewards"]),
        "metrics": totals, "seconds": time.perf_counter() - start,
    }
    return arrays, metadata


def validate_episode(data, gamma):
    count = len(data["rewards"])
    np.testing.assert_array_equal(data["periods"], np.arange(count))
    assert data["terminated"][-1] and np.count_nonzero(data["terminated"]) == 1
    # An independent one-step recursion check, without the target builder.
    expected = data["rewards"] + gamma * np.r_[data["targets"][1:], 0.0]
    np.testing.assert_allclose(data["targets"], expected, rtol=0, atol=1e-10)
    for key in ("obs", "calendar", "rewards", "targets", "uc_actions", "consumer_actions"):
        assert np.isfinite(data[key]).all(), key
    for i in (0, count // 2, count - 1):
        expected_time = public_calendar(datetime.fromisoformat(str(data["timestamps"][i])), i, count)
        np.testing.assert_array_equal(data["calendar"][i], expected_time)


def collect_data(output):
    contract, factory, bundle, old_value = load_context()
    before = actor_digest(bundle)
    protocol = {
        "source": str(SOURCE.relative_to(ROOT)), "actor_file_sha256": sha256(SOURCE / "actors/actors.pt"),
        "actor_parameter_sha256": before, "scenario": contract["scenario"], "gamma": contract["gamma"],
        "train_seeds": list(factory.seed_splits["train"]), "validation_seeds": list(VALIDATION_SEEDS),
        "test_loaded": False, "kinds": KINDS, "fit_seeds": FIT_SEEDS, "epochs": EPOCHS,
        "chunk_length": CHUNK_LENGTH, "chunks_per_batch": CHUNKS_PER_BATCH,
        "learning_rate": LR, "grad_clip": GRAD_CLIP, "value_scale": SCALE,
        "loss": "vf_loss_coeff(0.5) * 0.5 * mean(((prediction - full_terminal_MC)/20)^2)",
        "initialization": "fresh paired encoder/head per seed; time extra input weights zero; GRU new",
        "recurrent_inputs": "same 68 current UC observation fields; no extra actions, rewards or private data",
        "prefix": "replay from real month start at current weights before each minibatch, no prefix loss/gradient",
        "calendar_fields": CALENDAR_FIELDS, "sampling": "one frozen stochastic joint-policy trajectory per month",
        "evaluation": "final epoch fixed in advance; no early stopping or validation-based model selection",
        "source_sha256": source_hashes(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_before": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "runtime": {"torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads()},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "data").mkdir()
    write_json(output / "protocol.json", protocol)
    manifest = []
    for seed in (*factory.seed_splits["train"], *VALIDATION_SEEDS):
        arrays, metadata = collect_episode(
            factory, bundle, old_value, seed, contract["gamma"],
            progress=lambda s, n: print(json.dumps({"event": "collect_progress", "seed": s, "periods": n}), flush=True),
        )
        path = output / "data" / f"{seed}.npz"
        np.savez_compressed(path, **arrays)
        metadata.update(file=path.name, sha256=sha256(path))
        manifest.append(metadata)
        write_json(output / "data_manifest.json", manifest)
        print(json.dumps({"event": "collected", **metadata}), flush=True)
    after = actor_digest(bundle)
    assert before == after
    # Original validation used these exact seeds, CPU actors and true-month MC.
    original = json.loads((SOURCE / "validation-0003.json").read_text(encoding="utf-8"))["critic"]["modules"]["uc"]
    data = load_episodes(output, "validation", contract["gamma"])
    reference = statistics(data, [row["checkpoint_predictions"] for row in data])
    max_delta = max(abs(reference[group][key] - original[group][key])
                    for group in original for key in original[group])
    assert max_delta < 1e-5, (max_delta, reference, original)
    write_json(output / "collection_audit.json", {
        "actors_unchanged": before == after, "actor_parameter_sha256_after": after,
        "checkpoint_and_actor_export_match": True, "original_validation_max_abs_difference": max_delta,
        "train_periods": sum(row["periods"] for row in manifest if row["split"] == "train"),
        "validation_periods": sum(row["periods"] for row in manifest if row["split"] == "validation"),
        "true_terminals": len(manifest), "test_loaded": False,
    })


def load_episodes(output, split, gamma):
    manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
    result = []
    for item in manifest:
        if item["split"] != split:
            continue
        path = output / "data" / item["file"]
        assert sha256(path) == item["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            row = {key: archive[key] for key in archive.files}
        validate_episode(row, gamma)
        row["metadata"] = item
        result.append(row)
    return result


def statistics(episodes, predictions):
    return stratified_statistics(
        np.concatenate(predictions), np.concatenate([row["targets"] for row in episodes]),
        np.concatenate([row["periods"] for row in episodes]),
    )


def chunks_for(episodes, length=CHUNK_LENGTH):
    return [(i, start, min(start + length, len(row["targets"])))
            for i, row in enumerate(episodes) for start in range(0, len(row["targets"]), length)]


def schedule_for(chunks, seed, epochs=EPOCHS):
    rng = np.random.default_rng(seed + 100000)
    return [rng.permutation(len(chunks)).tolist() for _ in range(epochs)]


def tensors_for(episodes):
    return [{key: torch.as_tensor(row[key], dtype=torch.float32) for key in ("obs", "calendar", "targets")}
            for row in episodes]


def pack_batch(episodes, selected):
    parts = [{key: episodes[i][key][start:end] for key in ("obs", "calendar", "targets")}
             for i, start, end in selected]
    batch = {key: pad_sequence([part[key] for part in parts], batch_first=True)
             for key in ("obs", "calendar", "targets")}
    lengths = torch.tensor([end - start for _, start, end in selected])
    batch["mask"] = torch.arange(batch["obs"].shape[1])[None] < lengths[:, None]
    return batch


@torch.no_grad()
def evaluate_model(model, episodes, tensors):
    model.eval()
    predictions = [model.predict_episode(row["obs"], row["calendar"]).cpu().numpy() for row in tensors]
    return statistics(episodes, predictions), predictions


def train_one(output, kind, seed, train, validation):
    destination = output / f"{kind}-{seed}"
    destination.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    chunks = chunks_for(train)
    schedule = schedule_for(chunks, seed)
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    train_tensors, val_tensors = tensors_for(train), tensors_for(validation)
    model = CriticProbe(kind, train[0]["obs"].shape[-1], seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    parameters = sum(p.numel() for p in model.parameters())
    curve = []
    step, samples, prefix_rows, clipped = 0, 0, 0, 0
    maximum_gradient = 0.0
    started = time.perf_counter()
    for epoch in range(EPOCHS + 1):
        if epoch:
            model.train()
            for offset in range(0, len(chunks), CHUNKS_PER_BATCH):
                selected = [chunks[i] for i in schedule[epoch - 1][offset:offset + CHUNKS_PER_BATCH]]
                batch = pack_batch(train_tensors, selected)
                hidden = None
                if kind == "gru":
                    prefixes = [train_tensors[i]["obs"][:start] for i, start, _ in selected]
                    hidden = model.prefix_hidden(prefixes)
                    prefix_rows += sum(len(prefix) for prefix in prefixes)
                optimizer.zero_grad(set_to_none=True)
                predictions, _ = model(batch["obs"], batch["calendar"], hidden)
                errors = (predictions[batch["mask"]] - batch["targets"][batch["mask"]]) / SCALE
                loss = LOSS_COEFFICIENT * 0.5 * errors.square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
                maximum_gradient = max(maximum_gradient, float(norm))
                clipped += int(norm > GRAD_CLIP)
                optimizer.step()
                step += 1
                samples += int(batch["mask"].sum())
        train_metrics, _ = evaluate_model(model, train, train_tensors)
        val_metrics, predictions = evaluate_model(model, validation, val_tensors)
        row = {"epoch": epoch, "optimizer_steps": step, "loss_samples": samples,
               "train": train_metrics, "validation": val_metrics,
               "seconds": time.perf_counter() - started}
        curve.append(row)
        with (destination / "curve.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if epoch in (0, 1, 5, 10, 15, 20):
            print(json.dumps({"event": "fit", "kind": kind, "seed": seed, "epoch": epoch,
                              "train_rmse": train_metrics["all"]["rmse"],
                              "validation": val_metrics, "seconds": row["seconds"]}), flush=True)
    budget = {"optimizer_steps": step, "loss_samples": samples, "epochs": EPOCHS,
              "schedule_sha256": schedule_hash, "parameters": parameters, "prefix_forward_rows": prefix_rows,
              "gradient_clip_count": clipped, "max_preclip_gradient_norm": maximum_gradient}
    torch.save({"kind": kind, "seed": seed, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "budget": budget}, destination / "checkpoint.pt")
    # Validate checkpoint replay and recurrent context without touching trajectories.
    restored = CriticProbe(kind, train[0]["obs"].shape[-1], seed=seed)
    restored.load_state_dict(torch.load(destination / "checkpoint.pt", weights_only=True)["model"])
    _, replay = evaluate_model(restored, validation, val_tensors)
    for a, b in zip(predictions, replay, strict=True):
        np.testing.assert_array_equal(a, b)
    predictions_file = {str(row["metadata"]["seed"]): pred
                        for row, pred in zip(validation, predictions, strict=True)}
    np.savez_compressed(destination / "validation_predictions.npz", **predictions_file)
    result = {"kind": kind, "seed": seed, "budget": budget, "final": curve[-1],
              "initial": curve[0], "checkpoint_replay_exact": True,
              "per_month": {row["metadata"]["month"]: error_statistics(pred, row["targets"])
                            for row, pred in zip(validation, predictions, strict=True)}}
    write_json(destination / "result.json", result)
    return result


def fit_models(output):
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "collection_audit.json").read_text(encoding="utf-8"))
    assert audit["actors_unchanged"] and audit["true_terminals"] == 16
    assert source_hashes() == protocol["source_sha256"], "Experiment source changed after protocol registration"
    train = load_episodes(output, "train", protocol["gamma"])
    validation = load_episodes(output, "validation", protocol["gamma"])
    assert len(train) == 12 and len(validation) == 4
    results = []
    for seed in FIT_SEEDS:
        for kind in KINDS:
            result_path = output / f"{kind}-{seed}" / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = train_one(output, kind, seed, train, validation)
            results.append(result)
    for seed in FIT_SEEDS:
        peers = [r["budget"] for r in results if r["seed"] == seed]
        for key in ("optimizer_steps", "loss_samples", "epochs", "schedule_sha256"):
            assert all(p[key] == peers[0][key] for p in peers), key
    expected_samples = EPOCHS * sum(len(row["targets"]) for row in train)
    assert all(r["budget"]["loss_samples"] == expected_samples for r in results)
    groups = {}
    for kind in KINDS:
        peers = [r for r in results if r["kind"] == kind]
        groups[kind] = {
            split: {
                group: {metric: {
                    "mean": float(np.mean([r["final"][split][group][metric] for r in peers])),
                    "sd": float(np.std([r["final"][split][group][metric] for r in peers], ddof=1)),
                } for metric in ("rmse", "bias", "explained_variance")}
                for group in peers[0]["final"][split]
            } for split in ("train", "validation")
        }
    reference = statistics(validation, [row["checkpoint_predictions"] for row in validation])
    training_mean = float(np.concatenate([row["targets"] for row in train]).mean())
    constant = statistics(validation, [np.full_like(row["targets"], training_mean) for row in validation])
    report = {"protocol": protocol, "collection_audit": audit, "results": results, "groups": groups,
              "frozen_checkpoint_reference": reference, "training_mean_baseline": constant,
              "training_mean": training_mean, "budget_equal": True,
              "source_unchanged": source_hashes() == protocol["source_sha256"],
              "actors_file_unchanged": sha256(SOURCE / "actors/actors.pt") == protocol["actor_file_sha256"],
              "scope": "Conditional on one frozen joint policy and one stochastic path per month; seed SD measures optimization variability, not policy/environment generalization uncertainty."}
    assert report["source_unchanged"] and report["actors_file_unchanged"]
    write_json(output / "report.json", report)
    print(json.dumps({"event": "complete", "groups": groups, "budget_equal": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("all", "collect", "fit"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage in ("all", "collect"):
        collect_data(args.output)
    if args.stage in ("all", "fit"):
        fit_models(args.output)


if __name__ == "__main__":
    main()
