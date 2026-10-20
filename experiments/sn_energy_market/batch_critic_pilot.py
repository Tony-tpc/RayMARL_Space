"""Fixed-budget 24-batch pilot: separate Actor/Critic targets, no auxiliary fitting."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "RLAlgo/src"), str(ROOT / "environments/sn_energy_market/src")]

from experiments.sn_energy_market.evaluation import (
    GAMMA,
    VALIDATION_SEEDS,
    cursor,
    frozen_evaluation,
    retained_optimizer_steps,
    source_hashes,
)
from experiments.sn_energy_market.profile_dataset import DatasetEnvFactory, ProfileDataset
from experiments.sn_energy_market.train import model_settings, write_json
from experiments.sn_energy_market.training_audit import TrainingAudit

OUTPUT = ROOT / "outputs/ausgrid-batch-critic-pilot/20261020"
REFERENCE = ROOT / "outputs/ausgrid-online-critic-pilot/20261020/report.json"


def configuration(output):
    from raymarl.algorithms.stackelberg import StackelbergConfig

    dataset = ProfileDataset(
        ROOT / "datasets/ausgrid/processed/manifest.json",
        splits=("train", "validation"),
    )
    factory = DatasetEnvFactory(dataset, seed=42)
    return (
        StackelbergConfig()
        .scenarios(
            factory,
            train_seeds=factory.seed_splits["train"],
            validation_seeds=VALIDATION_SEEDS,
            test_seeds=(),
        )
        .debugging(seed=42)
        .rl_module(model_config=model_settings(32))
        .training(
            gamma=GAMMA,
            lambda_gae=0.9025,
            critic_lambda=1.0,
            critic_epochs=4,
            rollout_length=64,
            inner_updates=2,
            response_updates=2,
            uc_candidates=1,
            uc_updates_per_candidate=2,
            deviation_updates=0,
            num_epochs=2,
            minibatch_size=32,
            evaluation_steps=96,
            stopping_patience=0,
            storage_path=str(output),
            checkpoint_interval=0,
        )
    )


def run_pilot(output, reference_path):
    import ray
    import torch

    cfg = configuration(output)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    # Compare learning settings and data semantics; source hashes describe each
    # run separately and intentionally change when the implementation is cleaned.
    actual, expected = cfg.learning_contract(), reference["learning_contract"]
    for key, value in actual.items():
        if key not in {"critic_lambda", "scenario"}:
            assert value == expected[key], f"Pilot learning setting changed: {key}"
    for key in ("manifest_sha256", "loaded_splits", "household_ids", "config",
                "resource_calibration", "scenario_prices"):
        assert actual["scenario"][key] == expected["scenario"][key], key
    assert reference["ppo_sampled_periods"] == 1536
    assert reference["critic_auxiliary_periods"] == 0
    assert source_hashes("environments/sn_energy_market/src") == reference["environment_source_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    temporary = ROOT / ".tmp" / output.parent.name
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(temporary)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    token = ROOT / ".ray/auth_token"
    token.parent.mkdir(exist_ok=True)
    if not token.exists():
        token.write_text(secrets.token_hex(32), encoding="utf-8")
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(token)
    environment_before = source_hashes("environments/sn_energy_market/src")
    write_json(output / "experiment.json", {
        "learning_contract": cfg.learning_contract(),
        "initialization": "fresh_seed_42_same_as_reference",
        "reference_report": str(reference_path.relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "outer_rounds": 3,
        "ppo_budget": 1536,
        "critic_auxiliary_periods": 0,
        "actor_epochs": 2,
        "critic_epochs": 4,
        "minibatch_size": 32,
        "test_loaded": False,
    })
    ray.init(num_cpus=2, include_dashboard=False, _node_ip_address="127.0.0.1",
             _temp_dir=str(temporary / "ray"))
    algorithm = audit = None
    try:
        algorithm = cfg.build_algo()
        audit = TrainingAudit(algorithm, output).install()
        start = cursor(algorithm.env_runner)
        audit.phase = "validation_initial"
        initial = frozen_evaluation(algorithm)
        assert initial == reference["initial"], "Initial policies/evaluation differ from the baseline"
        assert start == reference["start"]
        rounds = []
        for index in range(3):
            audit.phase = f"round_{index + 1}"
            before = cursor(algorithm.env_runner)
            result = algorithm.train()
            row = {
                "round": algorithm.last_round,
                "learners": result.get("learners", {}),
                "start": before,
                "end": cursor(algorithm.env_runner),
            }
            assert row["end"]["main_periods"] - before["main_periods"] == 256
            rounds.append(row)
            write_json(output / f"round-{index + 1:04d}.json", row)
            print(json.dumps({"round": index + 1, "selected": algorithm.last_round["selected_candidate"],
                              "cursor": row["end"]}), flush=True)
        assert algorithm.sampled_periods == 1536
        assert cursor(algorithm.env_runner)["main_periods"] == 768
        assert len(audit.updates) == 48  # 24 batches, one critic and one actor phase each.
        from experiments.sn_energy_market.plot_batch_losses import load_batches

        batch_log = Path(algorithm.logdir) / "batch_losses.jsonl"
        batch_rows = load_batches(batch_log)
        assert [row["batch_id"] for row in batch_rows] == list(range(1, 25))
        assert all(row["samples"] == 64 and row["branch_status"] != "pending" for row in batch_rows)
        (output / "batch_losses.jsonl").write_bytes(batch_log.read_bytes())
        for key in algorithm.config.agent_order:
            steps = [row for row in audit.optimizer_steps if row["agent"] == key and row["role"] == "value"]
            assert len(steps) == 192
        audit.phase = "validation_final"
        final = frozen_evaluation(algorithm)
        audit.phase = "checkpoint_replay"
        saved = copy.deepcopy(algorithm.get_state())
        checkpoint = algorithm.save_to_path(str(output / "checkpoint"))
        expected = algorithm.env_runner.collect(8)
        algorithm.restore_from_path(checkpoint)
        actual = algorithm.env_runner.collect(8)
        for key, rows in expected.policy_batches.items():
            for name, values in rows.items():
                np.testing.assert_array_equal(values, actual[key][name])
        algorithm.set_state(saved)
        algorithm.export_actors(output / "actors")
        assert source_hashes("environments/sn_energy_market/src") == environment_before
        report = {
            "status": "completed",
            "initial": initial,
            "final": final,
            "start": start,
            "end": cursor(algorithm.env_runner),
            "rounds": rounds,
            "ppo_sampled_periods": 1536,
            "main_advanced_periods": 768,
            "critic_auxiliary_periods": 0,
            "critic_optimizer_steps_executed_per_module": 192,
            "initial_matches_reference": True,
            "reference_report": str(reference_path.relative_to(ROOT)),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "batch_loss_log": str((output / "batch_losses.jsonl").relative_to(ROOT)),
            "batch_loss_sha256": hashlib.sha256(batch_log.read_bytes()).hexdigest(),
            "logged_batches": 24,
            "checkpoint_exact_replay": True,
            "retained_optimizer_steps": [
                item.get() for item in algorithm.learner_group.foreach_learner(retained_optimizer_steps)
            ],
            "test_loaded": False,
            "test_evaluated": False,
            "environment_source_unchanged": True,
            "environment_source_sha256": environment_before,
            "learning_contract": cfg.learning_contract(),
            "diagnostics": audit.report(),
            "source_sha256": {
                **source_hashes("RLAlgo/src/raymarl/algorithms/stackelberg"),
                **source_hashes("experiments/sn_energy_market"),
            },
        }
        write_json(output / "report.json", report)
        write_json(output / "optimizer_steps.json", audit.optimizer_steps)
        print(json.dumps({"status": "completed", "report": str(output / "report.json")}), flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"error": repr(exc), "diagnostics": audit.report() if audit else None})
        raise
    finally:
        if audit is not None:
            audit.close()
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    run_pilot(OUTPUT, REFERENCE)
