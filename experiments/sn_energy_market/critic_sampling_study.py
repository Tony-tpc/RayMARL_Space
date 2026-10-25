"""D24 -> V9 always32: equivalence, fixed-data diagnosis, eight rounds, frozen evaluation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE = Path("D:/Codex/2026-10-26/raymarl-v9-critic")
STAGE = sys.argv[1] if len(sys.argv) > 1 else "preflight"
RUNTIME = BASE / "runtime" / STAGE
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = os.environ["TEMP"] = str(RUNTIME)
os.environ["RAY_AUTH_TOKEN_PATH"] = str(RUNTIME / "auth_token")
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import copy
import importlib.util
import json
import pickle
import random
import shutil
import subprocess
import time
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

from experiments.sn_energy_market.evaluation import (
    error_statistics,
    stratified_statistics,
    terminal_returns,
)
from experiments.sn_energy_market.rollback_comparison import (
    RollbackComparison,
    import_d_state,
)
from experiments.sn_energy_market.sampling_gate import ROOT
from experiments.sn_energy_market.sampling_gate.audit import ExperimentAudit
from experiments.sn_energy_market.sampling_gate.evaluate import (
    common_state_modes,
    complete_mc,
)
from experiments.sn_energy_market.sampling_gate.io import (
    read_json,
    source_manifest,
    verify_sources,
    write_json,
)
from experiments.sn_energy_market.sampling_gate.runtime import (
    EVALUATION_SEEDS,
    MONTH_SEEDS,
    configuration,
    digest_tree,
    ray_session,
)
from experiments.sn_energy_market.training_audit import reference_advantages
from raymarl.algorithms.stackelberg.runner import random_state

OUTPUT = BASE / "outputs" / "v9-always"
BUDGET = {"phase_sampling": "all_streams", "inner_updates": 4, "response_updates": 4,
          "uc_updates_per_candidate": 4, "critic_epochs": 2, "num_epochs": 1}
START, END = 24, 32


def config(*, mixed=True, output=OUTPUT):
    cfg = configuration(output, "D")
    cfg.algo_class = RollbackComparison
    cfg.initialization = "D24_full_learner_adam_sampling_state"
    cfg.always_keep_candidate = True
    if mixed:
        cfg.training(**BUDGET)
    return cfg


def seed_fit(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    random.seed(seed)


def load_d(algorithm, mixed=True):
    return import_d_state(algorithm, budget_changes=BUDGET if mixed else None)


def actor_hash(algorithm):
    return digest_tree({key: module.actor.state_dict()
                        for key, module in algorithm.learner_group._learner.module.items()})


def subset(batch, indices):
    return MultiAgentBatch({key: SampleBatch({name: values[indices] for name, values in data.items()})
                           for key, data in batch.policy_batches.items()}, len(indices))


def preflight():
    """Reference code is frozen before any V9 changes; no online optimizer work is retained."""
    reference = BASE / "backup" / "runner-before-v9.py"
    reference.write_bytes(subprocess.check_output([
        "git", "show", "27a0677:RLAlgo/src/raymarl/algorithms/stackelberg/runner.py"], cwd=ROOT))
    spec = importlib.util.spec_from_file_location("raymarl.algorithms.stackelberg._before_v9", reference)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    with ray_session(RUNTIME):
        algorithm = config(mixed=False, output=BASE / "verification" / "preflight-algorithm").build_algo()
        old_runner = None
        try:
            migration = load_d(algorithm, mixed=False)
            runner = algorithm.env_runner
            state = runner.get_state()
            initial_learner = digest_tree(algorithm.learner_group.get_state())
            old_runner = legacy.StackelbergEnvRunner(config=algorithm.config)
            old_runner.set_state(state)
            old_calls, new_calls = [], []
            original_old, original_new = old_runner._value, runner._values
            started = time.perf_counter()
            with patch.object(old_runner, "_value", side_effect=lambda *a: (old_calls.append(1), original_old(*a))[1]):
                expected = old_runner.collect(384)
            print(json.dumps({"preflight": "legacy_collected", "seconds": time.perf_counter() - started}), flush=True)
            runner.set_state(state)
            with patch.object(runner, "_values", side_effect=lambda *a: (new_calls.append(1), original_new(*a))[1]):
                actual = runner.collect(384)
            errors = {}
            operator_errors = {}
            numerical = {"vf_preds", "next_vf_preds", "advantages", "value_targets", "value_bootstrap"}
            for key, data in expected.policy_batches.items():
                observed = actual[key]
                residual_rows = {"rewards": np.zeros(384), "terminateds": data["terminateds"],
                    "segment_end": observed["segment_end"], "vf_preds": observed["vf_preds"] - data["vf_preds"],
                    "next_vf_preds": observed["next_vf_preds"] - data["next_vf_preds"]}
                propagated = reference_advantages(residual_rows, algorithm.config.gamma, algorithm.config.lambda_gae)
                # An independent TD sum must explain every GAE difference from float32 GEMV/GEMM.
                np.testing.assert_allclose(observed["advantages"] - data["advantages"], propagated, atol=1e-10, rtol=1e-10)
                with torch.no_grad():
                    network = copy.deepcopy(runner.module[key].value).double()
                    x = torch.as_tensor(data["critic_obs"], dtype=torch.float64)
                    reference_values = (network(x) * runner.module[key].value_scale).numpy()
                operator_errors[key] = {
                    "scalar_vs_float64_max_abs": float(np.max(np.abs(data["vf_preds"] - reference_values))),
                    "batch_vs_float64_max_abs": float(np.max(np.abs(observed["vf_preds"] - reference_values))),
                    "gae_difference_unexplained_max_abs": float(np.max(np.abs(observed["advantages"] - data["advantages"] - propagated))),
                }
                for name, values in data.items():
                    other = actual[key][name]
                    if name in numerical:
                        delta = np.abs(values - other)
                        errors[f"{key}/{name}"] = {"max_abs": float(delta.max()),
                            "max_relative": float(np.max(delta / np.maximum(np.abs(values), 1e-12))),
                            "strict_tolerance_passed": bool(np.allclose(other, values, atol=1e-5, rtol=1e-6))}
                        if name != "advantages":
                            np.testing.assert_allclose(other, values, atol=1e-5, rtol=1e-6)
                        else:
                            # Near-zero GAE has no relative scale; use measured value-error propagation,
                            # not an arbitrary larger absolute tolerance.
                            assert np.all(delta <= np.abs(propagated) + 1e-10)
                    else:
                        np.testing.assert_array_equal(other, values)
            for key in ("environment", "observation", "terminated", "data_exhausted", "scenario_seed", "action_rng", "rng", "sampled_periods"):
                assert digest_tree(old_runner.get_state()["sampling"][key]) == digest_tree(runner.get_state()["sampling"][key]), key
            assert digest_tree(algorithm.learner_group.get_state()) == initial_learner
            timings = {"scalar": [], "batched": []}
            for repeat in range(4):
                for name, current in (("scalar", old_runner), ("batched", runner)):
                    current.set_state(state)
                    started = time.perf_counter()
                    current.collect(384)
                    seconds = time.perf_counter() - started
                    if repeat:
                        timings[name].append(seconds)
                    print(json.dumps({"timing": name, "repeat": repeat, "seconds": seconds}), flush=True)
            runner.set_state(state)
            report = {"passed": True, "migration": migration, "exact_actions_rewards_physics_rng": True,
                      "parameters_adam_unchanged": True, "atol": 1e-5, "rtol": 1e-6,
                      "errors": errors, "critic_calls": {"scalar": len(old_calls), "batched": len(new_calls)},
                      "operator_errors": operator_errors,
                      "gae_acceptance": "exact independent propagation of measured FP32 value differences, residual <= 1e-10",
                      "collect_seconds": {name: {"samples": values, "median": float(np.median(values)),
                         "min": min(values), "max": max(values)} for name, values in timings.items()},
                      "sampling_device": str(next(runner.module["uc"].parameters()).device),
                      "learner_device": str(algorithm.learner_group._learner._device),
                      "source_sha256": source_manifest()}
            import psutil
            report["peak_process_bytes"] = psutil.Process().memory_info().peak_wset
            write_json(BASE / "verification" / "preflight.json", report)
        finally:
            if old_runner is not None:
                old_runner.stop()
            algorithm.stop()


def validation_cache(algorithm):
    """One D24 actor trajectory per paired validation seed; no MC label reaches training."""
    runner = algorithm.env_runner
    saved = runner.get_state()
    inputs, periods, months = {"uc": [], "consumer": []}, [], []
    targets = {key: [] for key in runner.agent_ids}
    try:
        for scenario in MONTH_SEEDS:
            for action_seed in EVALUATION_SEEDS:
                runner.restart(scenario)
                seed_fit(action_seed * 1_000_003 + scenario)
                runner._action_rng = random_state()
                batch = runner.collect(runner.env.engine.config.system.horizon_steps)
                assert batch["uc"]["terminateds"][-1]
                inputs["uc"].append(batch["uc"]["critic_obs"])
                inputs["consumer"].append(batch[runner.env.consumer_ids[0]]["critic_obs"])
                for key, data in batch.policy_batches.items():
                    targets[key].append(terminal_returns(data["rewards"], data["terminateds"], algorithm.config.gamma))
                periods.extend(range(batch.count))
                months.extend([scenario] * batch.count)
                print(json.dumps({"frozen_validation_cache": scenario, "action_seed": action_seed}), flush=True)
    finally:
        runner.set_state(saved)
    return {"inputs": {key: np.concatenate(value) for key, value in inputs.items()},
            "targets": {key: np.concatenate(value) for key, value in targets.items()},
            "periods": np.asarray(periods), "months": np.asarray(months)}


@torch.no_grad()
def score_values(algorithm, cache):
    learner = algorithm.learner_group._learner
    metrics = {}
    pooled_values, pooled_targets = [], []
    for key, module in learner.module.items():
        source = cache["inputs"]["uc" if key == "uc" else "consumer"]
        predictions = []
        for start in range(0, len(source), 2048):
            x = torch.as_tensor(source[start:start + 2048], dtype=torch.float32, device=learner._device)
            predictions.extend(module.compute_values({"obs": x, "critic_obs": x}).cpu().numpy())
        predictions = np.asarray(predictions)
        target = cache["targets"][key]
        metrics[key] = {**stratified_statistics(predictions, target, cache["periods"]),
                       "by_month": [{"scenario_seed": int(month), **error_statistics(
                           predictions[cache["months"] == month], target[cache["months"] == month])}
                           for month in MONTH_SEEDS]}
        if key != "uc":
            pooled_values.extend(predictions)
            pooled_targets.extend(target)
    metrics["consumer_pooled"] = error_statistics(pooled_values, pooled_targets)
    return metrics


def frozen():
    assert read_json(BASE / "verification" / "preflight.json")["passed"]
    with ray_session(RUNTIME):
        algorithm = config(output=BASE / "verification" / "frozen-algorithm").build_algo()
        try:
            migration = load_d(algorithm)
            state = algorithm.get_state()
            actors = actor_hash(algorithm)
            cache = validation_cache(algorithm)
            algorithm.set_state(state)
            batch = algorithm.env_runner.collect_phase(768)
            algorithm.set_state(state)
            initial = score_values(algorithm, cache)
            cache_dir = BASE / "outputs" / "frozen"
            cache_dir.mkdir(parents=True, exist_ok=True)
            with (cache_dir / "fixed-training-and-validation.pkl").open("wb") as handle:
                pickle.dump({"batch": batch, "validation": cache}, handle, protocol=5)
            data_hash = digest_tree(batch.policy_batches)
            rows = []
            for fit_seed in (11, 29, 47):
                for shift in range(4):
                    for organization in ("sequential", "mixed"):
                        if organization == "mixed" and shift:
                            continue
                        algorithm.set_state(state)
                        seed_fit(fit_seed)
                        learner = algorithm.learner_group._learner
                        before_steps = {name: max((int(s["step"]) for s in opt.state.values()), default=0)
                                        for name, opt in learner._named_optimizers.items()}
                        groups = ([(shift + i) % 4 for i in range(4)] if organization == "sequential" else [None])
                        for stream in groups:
                            selected = batch if stream is None else subset(batch, np.flatnonzero(batch["uc"]["stream_id"] == stream))
                            algorithm.learner_group.update(batch=selected, update_phase="critic", num_epochs=2,
                                                           minibatch_size=32, shuffle_batch_per_epoch=True)
                        after_steps = {name: max((int(s["step"]) for s in opt.state.values()), default=0)
                                       for name, opt in learner._named_optimizers.items()}
                        deltas = {name: after_steps[name] - before_steps[name] for name in before_steps}
                        assert all(delta == (48 if name.endswith("value") else 0) for name, delta in deltas.items()), deltas
                        assert actor_hash(algorithm) == actors
                        assert digest_tree(batch.policy_batches) == data_hash
                        row = {"organization": organization, "fit_seed": fit_seed, "shift": shift,
                               "optimizer_steps": deltas, "metrics": score_values(algorithm, cache)}
                        rows.append(row)
                        write_json(cache_dir / "progress.json", {"completed": len(rows), "total": 15, "last": row})
                        print(json.dumps({"frozen_fit": len(rows), "organization": organization,
                                          "seed": fit_seed, "shift": shift, "uc_rmse": row["metrics"]["uc"]["all"]["rmse"]}), flush=True)
            algorithm.set_state(state)
            restored = digest_tree(algorithm.get_state()) == digest_tree(state)
            assert restored, "Frozen diagnosis did not restore its starting state"
            write_json(cache_dir / "summary.json", {"passed": True, "migration": migration,
                "initial": initial, "trials": rows, "training_samples": 768, "epochs": 2,
                "critic_updates_each": 48, "actors_unchanged": True, "targets_fixed": True,
                "fixed_batch_sha256": data_hash, "online_state_restored": restored,
                "scope": "fixed-data order experiment; fit seeds are not independent online training seeds"})
        finally:
            algorithm.stop()


def train():
    assert read_json(BASE / "verification" / "preflight.json")["passed"]
    diagnosis = read_json(BASE / "outputs" / "frozen" / "summary.json")
    assert diagnosis["passed"] and diagnosis["online_state_restored"]
    if (OUTPUT / "status.json").exists():
        raise FileExistsError("An existing online run must not be overwritten or silently restarted")
    manifest = source_manifest()
    for name in manifest:
        target = OUTPUT / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    write_json(OUTPUT / "protocol.json", {"source_sha256": manifest, "start_round": START, "end_round": END,
        "phase_samples": 768, "stream_samples": 192, "critic_epochs": 2, "actor_epochs": 1,
        "ppo_periods": 24576, "committed_periods": 12288, "critic_steps_each": 1536,
        "uc_actor_steps_executed_and_retained": 192, "consumer_actor_steps_executed_each": 576,
        "primary_comparator": "historical always32", "secondary": ["D24", "gate32", "A24", "B24", "C24"],
        "same_optimizer_budget": True, "same_environment_budget": False,
        "decision_rule": {
            "primary_critic": "UC RMSE and absolute bias below historical always32; report D24 and every month",
            "empirical_guard": "No increase in worst-month UC RMSE, Consumer pooled RMSE/cost or native lost energy, and no decrease in UC return/cash",
            "tolerance": "zero observed worsening for a joint success label; this is not an approved business tolerance",
            "mixed_outcomes": "Report each metric without claiming overall success",
            "promote_default": False,
        },
        "validation_months": MONTH_SEEDS, "action_seeds": EVALUATION_SEEDS,
        "test_loaded": False, "mc_training_labels": 0})
    with ray_session(RUNTIME):
        algorithm = config().build_algo()
        audit = None
        profiling = ExitStack()
        seconds, calls = Counter(), Counter()

        def timed(name, function):
            def run(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    seconds[name] += time.perf_counter() - started
                    calls[name] += 1
            return run

        try:
            write_json(OUTPUT / "initialization.json", load_d(algorithm))
            audit = ExperimentAudit(algorithm, OUTPUT).install()
            for owner, method, name in (
                (algorithm, "_snapshot", "snapshot"), (algorithm, "_restore", "restore"),
                (algorithm, "_evaluate", "candidate_scoring_including_restore"),
                (algorithm.env_runner, "_period", "action_and_environment_periods"),
                (algorithm.env_runner, "_values", "batch_value_inference"),
                (algorithm.env_runner, "_bootstrap_inputs", "temporary_tail_announcement"),
            ):
                profiling.enter_context(patch.object(owner, method, timed(name, getattr(owner, method))))
            for n in range(START + 1, END + 1):
                verify_sources(manifest)
                write_json(OUTPUT / "status.json", {"status": "training", "completed_rounds": n - 1 - START,
                                                     "active_outer_round": n, "pid": os.getpid()})
                started = time.perf_counter()
                previous_seconds, previous_calls = seconds.copy(), calls.copy()
                algorithm.train()
                audit.assert_budget(n)
                assert algorithm.last_round["selected_candidate"] == 1
                record = {"round": algorithm.last_round, "gate_selected": algorithm.gate_selected,
                          "lanes": algorithm.env_runner.lane_cursors(), "seconds": time.perf_counter() - started,
                          "timing_seconds": dict(seconds - previous_seconds), "timing_calls": dict(calls - previous_calls)}
                import psutil
                record["peak_process_bytes"] = psutil.Process().memory_info().peak_wset
                write_json(OUTPUT / f"round-{n:04d}.json", record)
                saved = algorithm.save_to_path()
                write_json(OUTPUT / "checkpoint.json", {"path": saved, "outer_round": n})
                print(json.dumps({"completed_rounds": n - START, "outer_round": n, "seconds": record["seconds"]}), flush=True)
            algorithm.export_actors(OUTPUT / "actors")
            write_json(OUTPUT / "status.json", {"status": "trained", "completed_rounds": 8, "outer_round": END})
        except BaseException as exc:
            write_json(OUTPUT / "status.json", {"status": "failed", "completed_rounds": algorithm.outer_round - START,
                                                 "error": repr(exc)})
            raise
        finally:
            profiling.close()
            if audit is not None:
                audit.close()
            algorithm.stop()


def evaluate(kind):
    checkpoint = read_json(OUTPUT / "checkpoint.json")
    assert checkpoint["outer_round"] == END
    verify_sources(read_json(OUTPUT / "protocol.json")["source_sha256"])
    with ray_session(RUNTIME):
        algorithm = config().build_algo()
        try:
            algorithm.restore_from_path(checkpoint["path"])
            before = algorithm.get_state()
            first = algorithm.env_runner.collect_phase(32)
            algorithm.set_state(before)
            second = algorithm.env_runner.collect_phase(32)
            assert digest_tree(first.policy_batches) == digest_tree(second.policy_batches)
            algorithm.set_state(before)
            function = complete_mc if kind == "evaluate" else common_state_modes
            folder = "validation" if kind == "evaluate" else "mode-evaluation"
            write_json(OUTPUT / "status.json", {"status": "evaluating", "phase": kind, "completed_rounds": 8})
            function(algorithm, OUTPUT / folder)
            assert digest_tree(before) == digest_tree(algorithm.get_state())
            both = all((OUTPUT / folder / "summary.json").exists() for folder in ("validation", "mode-evaluation"))
            write_json(OUTPUT / "status.json", {"status": "completed" if both else "evaluated_mc",
                "phase": kind, "completed_rounds": 8, "checkpoint_replay_exact": True})
        finally:
            algorithm.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "frozen", "train", "evaluate", "modes"))
    args = parser.parse_args()
    if args.stage in ("evaluate", "modes"):
        evaluate(args.stage)
    else:
        {"preflight": preflight, "frozen": frozen, "train": train}[args.stage]()


if __name__ == "__main__":
    main()
