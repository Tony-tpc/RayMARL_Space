"""Fixed-budget training and validation: separate Actor/Critic targets, no auxiliary fitting."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import sys
import time
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
from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)
from experiments.sn_energy_market.train import (
    inventory_valued_config,
    model_settings,
    write_json,
)
from experiments.sn_energy_market.training_audit import TrainingAudit

OUTPUT = ROOT / "outputs/ausgrid-hybrid-storage/20261022-phase-batch-pilot"
REFERENCE = ROOT / "outputs/ausgrid-batch-critic-pilot/20261020/report.json"
FULL_MONTH_REFERENCE = ROOT / "outputs/ausgrid-year-study/20261020/validation-0003.json"


def configuration(output):
    from raymarl.algorithms.stackelberg import StackelbergConfig

    dataset = ProfileDataset(
        ROOT / "datasets/ausgrid/processed/manifest.json",
        splits=("train", "validation"),
    )
    factory = DatasetEnvFactory(dataset, inventory_valued_config(dataset.environment_config()), seed=42)
    return (
        StackelbergConfig()
        .scenarios(
            factory,
            train_seeds=factory.seed_splits["train"],
            train_seed_groups=factory.training_seed_groups,
            validation_seeds=VALIDATION_SEEDS,
            test_seeds=(),
        )
        .debugging(seed=42)
        .rl_module(model_config=model_settings(32))
        .training(
            gamma=GAMMA,
            lambda_gae=0.97,
            critic_lambda=1.0,
            critic_epochs=4,
            rollout_length=192,
            inner_updates=2,
            response_updates=2,
            uc_candidates=1,
            uc_updates_per_candidate=2,
            deviation_updates=0,
            num_epochs=2,
            minibatch_size=32,
            evaluation_steps=None,
            stopping_patience=0,
            storage_path=str(output),
            checkpoint_interval=0,
        )
    )


def round_budget(cfg):
    """Actual branch work and retained main trajectory for this fixed pilot."""
    if cfg.deviation_updates:
        raise ValueError("Fixed pilot budget requires deviation_updates=0")
    phase_units = ([cfg.inner_updates] + [cfg.uc_updates_per_candidate] * cfg.uc_candidates
                   + [cfg.response_updates] * (cfg.uc_candidates + 1))
    sizes = [units * cfg.rollout_length for units in phase_units]
    minibatches = sum((size + cfg.minibatch_size - 1) // cfg.minibatch_size for size in sizes)
    return {
        "batches": len(sizes),
        "ppo_periods": sum(sizes),
        "main_periods": (cfg.inner_updates + cfg.response_updates) * cfg.rollout_length,
        "critic_steps_per_module": cfg.critic_epochs * minibatches,
    }


def run_pilot(output, reference_path, *, outer_rounds=3, observer=None, checkpoint_each_round=False,
              recovery=None, evaluation_rounds=None, full_period_checks=True):
    import ray
    import torch

    cfg = configuration(output)
    if recovery is not None and hasattr(recovery, "configure"):
        cfg = recovery.configure(cfg)
    budget = round_budget(cfg)
    evaluation_rounds = sorted(set(evaluation_rounds if evaluation_rounds is not None else
                                  (n for n in (3, 12, 24, 48, 72, outer_rounds) if n <= outer_rounds)))
    assert outer_rounds in evaluation_rounds and all(0 < n <= outer_rounds for n in evaluation_rounds)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    full_reference = json.loads(FULL_MONTH_REFERENCE.read_text(encoding="utf-8"))
    assert full_reference["performance"] == reference["final"]["performance"]
    assert full_reference["critic"]["modules"] == reference["final"]["critic"]["modules"]
    # Historical reports retain their original action/reward/evaluation contracts.
    # They supply descriptive reference results, not an identical-objective control.
    if recovery is None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        recovery.verify(cfg, outer_rounds)
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
    execution_sources = {
        **source_hashes("RLAlgo/src/raymarl/algorithms/stackelberg"),
        **source_hashes("experiments/sn_energy_market"),
    }
    write_json((output if recovery is None else recovery.folder) / "experiment.json", {
        "source_sha256": execution_sources,
        "learning_contract": cfg.learning_contract(),
        "initialization": "fresh_seed_42_hybrid_storage_full_episode_inventory_objective",
        "reference_report": str(reference_path.relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "outer_rounds": outer_rounds,
        "ppo_budget": budget["ppo_periods"] * outer_rounds,
        "round_budget": budget,
        "evaluation_rounds": evaluation_rounds,
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
        if recovery is not None:
            recovery.restore(algorithm)
            cfg = algorithm.config
            metadata_path = recovery.folder / "experiment.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(learning_contract=cfg.learning_contract(), initialization="checkpoint_restore")
            write_json(metadata_path, metadata)
        audit = TrainingAudit(algorithm, output, full_period_checks=full_period_checks).install()
        if observer is not None:
            observer.on_ready(algorithm, audit)
        if recovery is None:
            start = cursor(algorithm.env_runner)
            audit.phase = "validation_initial"
            if observer is not None:
                observer.before_evaluation(0)
            initial = frozen_evaluation(algorithm)
            write_json(output / "validation-0000.json", initial)
            if observer is not None:
                observer.after_evaluation(0, initial)
            evaluations, rounds, first_round = {"0": initial}, [], 0
        else:
            evaluations, rounds = recovery.history()
            initial, start, first_round = evaluations["0"], rounds[0]["start"], recovery.round
        started = time.monotonic()
        for index in range(first_round, outer_rounds):
            audit.phase = f"round_{index + 1}"
            before = cursor(algorithm.env_runner)
            result = algorithm.train()
            row = {
                "round": algorithm.last_round,
                "learners": result.get("learners", {}),
                "start": before,
                "end": cursor(algorithm.env_runner),
            }
            assert row["end"]["main_periods"] - before["main_periods"] == budget["main_periods"]
            rounds.append(row)
            write_json(output / f"round-{index + 1:04d}.json", row)
            if checkpoint_each_round:
                algorithm.save_to_path()
            if observer is not None:
                observer.after_round(index + 1, row)
            print(json.dumps({"round": index + 1, "selected": algorithm.last_round["selected_candidate"],
                              "cursor": row["end"], "elapsed_seconds": time.monotonic() - started}), flush=True)
            if index + 1 in evaluation_rounds:
                audit.phase = f"validation_{index + 1}"
                if observer is not None:
                    observer.before_evaluation(index + 1)
                evaluation = frozen_evaluation(algorithm)
                evaluations[str(index + 1)] = evaluation
                write_json(output / f"validation-{index + 1:04d}.json", evaluation)
                # Managed checkpoints keep the latest two, including their full
                # environment, optimizer and RNG state.
                if not checkpoint_each_round:
                    algorithm.save_to_path()
                if observer is not None:
                    observer.after_evaluation(index + 1, evaluation)
                print(json.dumps({"validated_round": index + 1,
                                  "uc_rmse": evaluation["critic"]["modules"]["uc"]["all"]["rmse"],
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
        assert algorithm.sampled_periods == budget["ppo_periods"] * outer_rounds
        main_periods = budget["main_periods"] * outer_rounds
        assert cursor(algorithm.env_runner)["main_periods"] == main_periods
        assert len(audit.updates) == 2 * budget["batches"] * (outer_rounds - first_round)
        from experiments.sn_energy_market.plot_batch_losses import load_batches

        batch_log = Path(algorithm.logdir) / "batch_losses.jsonl"
        batch_rows = load_batches(batch_log)
        assert [row["batch_id"] for row in batch_rows] == list(range(1, budget["batches"] * outer_rounds + 1))
        phase_units = {"consumer_adaptation": cfg.inner_updates, "uc_proposal": cfg.uc_updates_per_candidate,
                       "response": cfg.response_updates}
        assert all(row["samples"] == cfg.rollout_length * phase_units[row["phase"]]
                   and row["branch_status"] != "pending" for row in batch_rows)
        (output / "batch_losses.jsonl").write_bytes(batch_log.read_bytes())
        for key in algorithm.config.agent_order:
            steps = [row for row in audit.optimizer_steps if row["agent"] == key and row["role"] == "value"]
            assert len(steps) == budget["critic_steps_per_module"] * (outer_rounds - first_round)
        final = evaluations[str(outer_rounds)]
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
            "outer_rounds": outer_rounds,
            "evaluations": evaluations,
            "elapsed_training_and_validation_seconds": time.monotonic() - started,
            "ppo_sampled_periods": budget["ppo_periods"] * outer_rounds,
            "main_advanced_periods": main_periods,
            "critic_auxiliary_periods": 0,
            "critic_optimizer_steps_executed_per_module": budget["critic_steps_per_module"] * outer_rounds,
            "initial_actor_performance_matches_reference": False,
            "initial_critic_matches_reference": False,
            "reference_report": str(reference_path.relative_to(ROOT)),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "full_month_reference": str(FULL_MONTH_REFERENCE.relative_to(ROOT)),
            "full_month_reference_sha256": hashlib.sha256(FULL_MONTH_REFERENCE.read_bytes()).hexdigest(),
            "batch_loss_log": str((output / "batch_losses.jsonl").relative_to(ROOT)),
            "batch_loss_sha256": hashlib.sha256(batch_log.read_bytes()).hexdigest(),
            "logged_batches": budget["batches"] * outer_rounds,
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
            "source_sha256": execution_sources,
        }
        if recovery is not None:
            recovery.finish(report, audit)
        write_json(output / "report.json", report)
        write_json(output / "optimizer_steps.json", audit.optimizer_steps)
        if observer is not None:
            observer.on_completed(report)
        print(json.dumps({"status": "completed", "report": str(output / "report.json")}), flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"error": repr(exc), "diagnostics": audit.report() if audit else None})
        if observer is not None:
            observer.on_failure(exc)
        raise
    finally:
        if observer is not None:
            observer.close()
        if audit is not None:
            audit.close()
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    run_pilot(OUTPUT, REFERENCE)
