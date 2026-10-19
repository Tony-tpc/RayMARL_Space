"""Two fixed-budget, checkpoint-connected pilots; training and validation only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "RLAlgo/src"),
    str(ROOT / "environments/sn_energy_market/src"),
]

from experiments.sn_energy_market.calibrated_pilot import (
    GAMMA,
    collect_value_data,
    evaluate,
    source_hashes,
    terminal_returns,
)
from experiments.sn_energy_market.pilot import (
    TrainingAudit,
    assert_same_state,
    parameter_changes,
)
from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)
from experiments.sn_energy_market.train import model_settings, write_json

OUTPUT = ROOT / "outputs/ausgrid-continuous-pilot/20261019"
VALIDATION_SEEDS = (10042, 10045, 10048, 10051)


def configuration(output):
    from raymarl.algorithms.stackelberg import StackelbergConfig

    dataset = ProfileDataset(
        ROOT / "datasets/ausgrid/processed/manifest.json",
        splits=("train", "validation"),
    )
    factory = DatasetEnvFactory(dataset, seed=42, training_start_days=(1,))
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


def cursor(runner):
    state = runner.env.engine.state
    return {
        "seed": runner._scenario_seed,
        "period": state.period_id,
        "timestamp": state.timestamp.isoformat(),
        "terminated": runner._terminated,
        "main_periods": runner.trajectory_periods,
        "soc_kwh": state.ess_energy_kwh,
        "monthly_uc_kwh": sum(c.monthly_uc_energy_kwh for c in state.consumers),
        "tasks": sum(len(c.tasks) for c in state.consumers),
    }


def error_statistics(predictions, targets):
    error = np.asarray(predictions) - np.asarray(targets)
    variance = np.var(targets)
    return {
        "samples": len(error),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "prediction_mean": float(np.mean(predictions)),
        "return_mean": float(np.mean(targets)),
        "explained_variance": float(1 - np.var(error) / variance)
        if variance > 0
        else 0.0,
    }


def stratified_statistics(predictions, targets, periods):
    days = np.asarray(periods) // 48 + 1
    return {
        name: error_statistics(np.asarray(predictions)[mask], np.asarray(targets)[mask])
        for name, mask in {
            "all": days > 0,
            "early_1_10": days <= 10,
            "middle_11_20": (days > 10) & (days <= 20),
            "late_21_end": days > 20,
        }.items()
        if mask.any()
    }


def full_value_validation(algorithm):
    """One frozen trajectory per validation month, every state's true-terminal MC."""
    runner = algorithm.env_runner
    saved = runner.get_state(components="sampling")
    predictions = {key: [] for key in runner.agent_ids}
    targets = {key: [] for key in runner.agent_ids}
    period_ids, matched, episodes = [], [], []
    try:
        for seed in VALIDATION_SEEDS:
            runner.restart(seed)
            rows = {key: [] for key in runner.agent_ids}
            while not runner._terminated:
                samples, _ = runner._period()
                for key, samples_for_agent in rows.items():
                    samples_for_agent.append(samples[key])
            periods = [row["period_id"] for row in rows["uc"]]
            period_ids.extend(periods)
            episodes.append(
                {"seed": seed, "periods": len(periods), "true_terminal": True}
            )
            for key, samples in rows.items():
                values = [row["vf_preds"] for row in samples]
                returns = terminal_returns(
                    np.array([row["rewards"] for row in samples]),
                    np.array([row["terminateds"] for row in samples]),
                    GAMMA,
                )
                predictions[key].extend(values)
                targets[key].extend(returns)
                if key == "uc":
                    matched.append(
                        {
                            "seed": seed,
                            "period": 96,
                            "prediction": values[96],
                            "full_terminal_return": float(returns[96]),
                            "sample_error": float(values[96] - returns[96]),
                        }
                    )
            print(
                json.dumps({"validation_month": seed, "periods": len(periods)}),
                flush=True,
            )
    finally:
        runner.set_state(saved)
    return {
        "episodes": episodes,
        "periods": len(period_ids),
        "modules": {
            key: stratified_statistics(predictions[key], targets[key], period_ids)
            for key in predictions
        },
        "matched_period_96": matched,
        "matched_period_96_rmse": float(
            np.sqrt(np.mean([r["sample_error"] ** 2 for r in matched]))
        ),
        "scope": "Single frozen stochastic MC trajectory per month; correlated samples, not exact expectations.",
    }


def fixed_fit_statistics(algorithm, data):
    import torch

    result = {}
    for key, rows in data.items():
        module = algorithm.env_runner.module[key]
        with torch.no_grad():
            values = (
                module.compute_values(
                    {
                        name: torch.as_tensor(rows[name], dtype=torch.float32)
                        for name in ("obs", "critic_obs")
                    }
                )
                .cpu()
                .numpy()
            )
        result[key] = stratified_statistics(
            values, rows["value_targets"], np.arange(len(values))
        )
    return result


def retained_optimizer_steps(learner):
    return {
        name: sorted(
            {
                int(state["step"].item())
                for state in optimizer.state.values()
                if "step" in state
            }
        )
        for name, optimizer in learner._named_optimizers.items()
    }


def frozen_evaluation(algorithm):
    before = copy.deepcopy(algorithm.learner_group.get_state())
    result = {
        "performance": evaluate(algorithm, VALIDATION_SEEDS),
        "critic": full_value_validation(algorithm),
    }
    # Includes every optimizer, not only policy weights.
    assert_same_state(before, algorithm.learner_group.get_state())
    return result


def month_boundary_probe(algorithm):
    """Frozen continuation from the current main state across one real month end."""
    runner = algorithm.env_runner
    saved = runner.get_state(components="sampling")
    start = cursor(runner)
    try:
        remaining = (
            runner.env.config.system.horizon_steps - runner.env.engine.state.period_id
        )
        batch = runner.collect(remaining + 1)
        uc = batch["uc"]
        terminal_indices = np.flatnonzero(uc["terminateds"])
        np.testing.assert_array_equal(terminal_indices, [remaining - 1])
        assert uc["next_vf_preds"][remaining - 1] == 0
        assert uc["period_id"][-1] == 0
        assert uc["scenario_seed"][-1] != start["seed"]
        return {
            "start": start,
            "end": cursor(runner),
            "periods": batch.count,
            "terminal_rows": 1,
            "terminal_bootstrap_zero": True,
            "gradient_updates": 0,
        }
    finally:
        runner.set_state(saved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    args = parser.parse_args(argv)
    output = OUTPUT / f"stage-{args.stage}"
    output.mkdir(parents=True, exist_ok=False)

    import ray
    import torch
    from raymarl.algorithms.stackelberg.runner import random_state, restore_random_state

    torch.set_num_threads(1)
    temporary = ROOT / ".tmp/ausgrid-continuous-pilot"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(temporary)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    token = ROOT / ".ray/auth_token"
    token.parent.mkdir(exist_ok=True)
    if not token.exists():
        token.write_text(secrets.token_hex(32), encoding="utf-8")
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(token)
    environment_before = source_hashes("environments/sn_energy_market/src")
    cfg = configuration(output)
    write_json(
        output / "experiment.json",
        {
            "learning_contract": cfg.learning_contract(),
            "stage": args.stage,
            "ppo_budget": 1536,
            "critic_mc_budget": 1488,
            "critic_epochs": 32,
            "test_loaded": False,
        },
    )
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(temporary / "ray"),
    )
    algorithm = audit = None
    try:
        algorithm = cfg.build_algo()
        if args.stage == 2:
            algorithm.restore_from_path(str(OUTPUT / "stage-1/checkpoint"))
        audit = TrainingAudit(algorithm, output).install()
        start = cursor(algorithm.env_runner)
        sampled_before = algorithm.sampled_periods
        audit.phase = "validation_initial"
        initial_critic_source = None
        if args.stage == 2:
            previous_path = OUTPUT / "stage-1/report.json"
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            assert start == previous["end"]
            initial = {
                "performance": evaluate(algorithm, VALIDATION_SEEDS),
                "critic": previous["final"]["critic"],
            }
            # Check restored policies and short-window bootstrap values exactly.
            # The same checkpoint's full-month Critic baseline is already measured.
            assert initial["performance"] == previous["final"]["performance"]
            initial_critic_source = {
                "report": str(previous_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                "field": "final.critic",
                "new_full_month_periods": 0,
            }
        else:
            initial = frozen_evaluation(algorithm)
        sampling_before = algorithm.env_runner.get_state(components="sampling")
        weights_before = copy.deepcopy(algorithm.learner_group.get_weights())
        optimizer_before = [
            item.get()
            for item in algorithm.learner_group.foreach_learner(
                retained_optimizer_steps
            )
        ]
        data, episodes = collect_value_data(algorithm, (42,), audit)
        fit_before = fixed_fit_statistics(algorithm, data)
        audit.phase = "critic_prefit"
        rng = random_state()
        try:
            fitted = [
                item.get()
                for item in algorithm.learner_group.foreach_learner(
                    lambda learner, batches=data: learner.fit_values(
                        batches, epochs=32, minibatch_size=64
                    )
                )
            ]
            algorithm._sync()
        finally:
            restore_random_state(rng)
        fit_after = fixed_fit_statistics(algorithm, data)
        changes = parameter_changes(
            weights_before, algorithm.learner_group.get_weights()
        )
        assert all(change["actor"] == 0 for change in changes.values())
        after_sampling = algorithm.env_runner.get_state(components="sampling")
        assert (
            sampling_before["sampling"]["environment"].engine.state
            == after_sampling["sampling"]["environment"].engine.state
        )
        assert_same_state(
            sampling_before["sampling"]["rng"], after_sampling["sampling"]["rng"]
        )
        assert cursor(algorithm.env_runner) == start
        optimizer_fit = [
            item.get()
            for item in algorithm.learner_group.foreach_learner(
                retained_optimizer_steps
            )
        ]
        for before, after in zip(optimizer_before, optimizer_fit, strict=True):
            for name in before:
                if name.endswith("actor"):
                    assert before[name] == after[name]
        write_json(
            output / "critic_fit.json",
            {
                "fits": fitted,
                "episodes": episodes,
                "before": fit_before,
                "after": fit_after,
                "actor_unchanged": True,
                "sampling_rng_restored": True,
                "parameter_changes": changes,
            },
        )
        del data
        rounds = []
        for index in range(3):
            audit.phase = f"round_{algorithm.outer_round + 1}"
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
            print(
                json.dumps(
                    {
                        "round": algorithm.outer_round,
                        "selected": algorithm.last_round["selected_candidate"],
                        "cursor": row["end"],
                    }
                ),
                flush=True,
            )
        assert algorithm.sampled_periods - sampled_before == 1536
        audit.phase = "validation_final"
        final = frozen_evaluation(algorithm)
        audit.phase = "checkpoint_replay"
        saved = algorithm.get_state()
        checkpoint = algorithm.save_to_path(str(output / "checkpoint"))
        expected = algorithm.env_runner.collect(8)
        algorithm.restore_from_path(checkpoint)
        actual = algorithm.env_runner.collect(8)
        for key, rows in expected.policy_batches.items():
            for name, values in rows.items():
                np.testing.assert_array_equal(values, actual[key][name])
        algorithm.set_state(saved)
        # The first pilot checks the remaining July boundary without training;
        # the second actually reaches it with the selected training branches.
        audit.phase = "frozen_boundary_probe"
        boundary = month_boundary_probe(algorithm) if args.stage == 1 else None
        assert cursor(algorithm.env_runner)["main_periods"] == args.stage * 768
        if args.stage == 2:
            assert (
                cursor(algorithm.env_runner)["timestamp"] == "2010-08-02T00:00:00+10:00"
            )
        algorithm.export_actors(output / "actors")
        assert source_hashes("environments/sn_energy_market/src") == environment_before
        # Current episode records belong to the committed branch only.
        tier_counts = Counter()
        for record in algorithm.env_runner.env.engine.records:
            for consumer in record.decision_state.consumers:
                tier = sum(
                    consumer.monthly_uc_energy_kwh >= threshold
                    for threshold in algorithm.env_runner.env.config.market.tier_thresholds_kwh
                )
                tier_counts[str(tier)] += 1
        report = {
            "status": "passed",
            "stage": args.stage,
            "initial": initial,
            "initial_critic_source": initial_critic_source,
            "final": final,
            "start": start,
            "end": cursor(algorithm.env_runner),
            "rounds": rounds,
            "ppo_sampled_periods": 1536,
            "main_advanced_periods": 768,
            "critic_sampled_periods": sum(row["periods"] for row in episodes),
            "critic_fit": fitted,
            "critic_strata_before": fit_before,
            "critic_strata_after": fit_after,
            "checkpoint_exact_replay": True,
            "frozen_boundary_probe": boundary,
            "current_committed_episode_tier_counts": dict(tier_counts),
            "retained_optimizer_steps": [
                item.get()
                for item in algorithm.learner_group.foreach_learner(
                    retained_optimizer_steps
                )
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
        print(
            json.dumps({"status": "passed", "report": str(output / "report.json")}),
            flush=True,
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {"error": repr(exc), "diagnostics": audit.report() if audit else None},
        )
        raise
    finally:
        if audit is not None:
            audit.close()
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
