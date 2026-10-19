"""Small train-only pilot with diverse dates, explicit priors and value prefit."""

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
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "RLAlgo/src"),
    str(ROOT / "environments/sn_energy_market/src"),
]

from experiments.sn_energy_market.pilot import TrainingAudit, parameter_changes
from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)
from experiments.sn_energy_market.train import (
    model_settings,
    write_json,
)

OUTPUT = ROOT / "outputs/ausgrid-calibrated-pilot/20261019"
GAMMA = 0.9801


def terminal_returns(rewards, terminated, gamma):
    """Original reward units, complete finite episodes, no synthetic terminal."""
    assert terminated[-1]
    result = np.empty_like(rewards, dtype=np.float64)
    carry = 0.0
    for index in reversed(range(len(rewards))):
        carry = float(rewards[index]) + gamma * carry * (not terminated[index])
        result[index] = carry
    return result


def collect_value_data(algorithm, seeds, audit):
    accumulated = {key: [] for key in algorithm.config.agent_order}
    trajectories = []
    runner = algorithm.env_runner
    if not seeds or not set(seeds) <= set(algorithm.config.train_seeds):
        raise ValueError("Value fitting data requires nonempty training seeds only")
    saved = runner.get_state(components="sampling")
    phase = audit.phase
    try:
        for seed in seeds:
            audit.phase = "critic_collect"
            runner.restart(seed)
            start = runner.env.engine.state.timestamp.isoformat()
            steps = runner.env.config.system.horizon_steps - runner.env.engine.state.period_id
            batch = runner.collect(steps)
            assert runner._terminated
            for key, rows in batch.policy_batches.items():
                accumulated[key].append(
                    {
                        "obs": rows["obs"],
                        "critic_obs": rows["critic_obs"],
                        "value_targets": terminal_returns(
                            rows["rewards"], rows["terminateds"], algorithm.config.gamma
                        ),
                    }
                )
            trajectories.append(
                {"seed": seed, "start": start, "periods": steps, "true_terminal": True}
            )
            print(json.dumps({"critic_episode": seed, "periods": steps}), flush=True)
    finally:
        runner.set_state(saved)
        audit.phase = phase
    merged = {
        key: {name: np.concatenate([row[name] for row in rows]) for name in rows[0]}
        for key, rows in accumulated.items()
    }
    return merged, trajectories


def value_calibration(algorithm, seeds):
    """Same-state frozen validation samples; never used as fitting targets."""
    runner = algorithm.env_runner
    saved = runner.get_state()
    rows = []
    try:
        for seed in seeds:
            runner.restart(seed)
            for _ in range(96):
                runner._period()
            prediction = runner._value("uc", runner._observation, runner._observation)
            total = 0.0
            for t in range(96):
                _, transition = runner._period()
                total += GAMMA**t * transition.rewards["uc"]
            rows.append(
                {
                    "seed": seed,
                    "prediction": prediction,
                    "following_96_step_return": total,
                    "sample_error": prediction - total,
                }
            )
    finally:
        runner.set_state(saved)
    return {
        "rows": rows,
        "truncated_sample_rmse": float(
            np.sqrt(np.mean([row["sample_error"] ** 2 for row in rows]))
        ),
        "note": "Single truncated MC sample per state; diagnostic, not exact expected value.",
    }


def evaluate(algorithm, seeds):
    before = copy.deepcopy(algorithm.learner_group.get_weights())
    result = {
        "sampled": algorithm.evaluate_pair(seeds),
        "mean": algorithm.evaluate_pair(seeds, deterministic=True),
    }
    after = algorithm.learner_group.get_weights()
    assert all(
        value == 0
        for roles in parameter_changes(before, after).values()
        for value in roles.values()
    )
    return result


def source_hashes(directory):
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / directory).rglob("*.py"))
    }


def main():
    import ray
    import torch
    from raymarl.algorithms.stackelberg import StackelbergConfig

    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    environment_before = source_hashes("environments/sn_energy_market/src")
    dataset = ProfileDataset(
        ROOT / "datasets/ausgrid/processed/manifest.json",
        splits=("train", "validation"),
    )
    factory = DatasetEnvFactory(
        dataset, seed=42, training_start_days=(1, 11, 21, -1)
    )
    windows = factory.get_contract()["seed_windows"]
    selected = (("2010-07", 11), ("2010-10", 21), ("2011-01", 31))
    train_seeds = tuple(
        next(
            int(seed)
            for seed, window in windows.items()
            if (window["name"], window["start_day"]) == target
            and window["split"] == "train"
        )
        for target in selected
    )
    validation_seeds = (10042, 10045, 10048, 10051)
    configuration = (
        StackelbergConfig()
        .scenarios(
            factory,
            train_seeds=train_seeds,
            validation_seeds=validation_seeds,
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
            storage_path=str(OUTPUT),
            checkpoint_interval=0,
        )
    )
    write_json(
        OUTPUT / "experiment.json",
        {
            "learning_contract": configuration.learning_contract(),
            "selected_dates": selected,
            "critic_prefit_epochs": 32,
            "test_loaded": False,
            "formal_training": False,
        },
    )
    temporary = ROOT / ".tmp/ausgrid-calibrated-pilot"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(temporary)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    token = ROOT / ".ray/auth_token"
    token.parent.mkdir(exist_ok=True)
    if not token.exists():
        token.write_text(secrets.token_hex(32), encoding="utf-8")
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(token)
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(temporary / "ray"),
    )
    algorithm, audit = None, None
    try:
        algorithm = configuration.build_algo()
        audit = TrainingAudit(algorithm, OUTPUT).install()
        initial_weights = copy.deepcopy(algorithm.learner_group.get_weights())
        audit.phase = "validation_initial"
        initial = evaluate(algorithm, validation_seeds)
        initial_value = value_calibration(algorithm, validation_seeds)
        data, critic_episodes = collect_value_data(algorithm, train_seeds, audit)
        audit.phase = "critic_prefit"
        fitted = [
            item.get()
            for item in algorithm.learner_group.foreach_learner(
                lambda learner, batches=data: learner.fit_values(batches, epochs=32, minibatch_size=64)
            )
        ]
        algorithm._sync()
        fit_changes = parameter_changes(
            initial_weights, algorithm.learner_group.get_weights()
        )
        assert all(change["actor"] == 0 for change in fit_changes.values())
        write_json(
            OUTPUT / "critic_fit.json",
            {
                "fits": fitted,
                "episodes": critic_episodes,
                "parameter_changes": fit_changes,
            },
        )
        print(json.dumps({"critic_fit": fitted[0]["modules"]["uc"]}), flush=True)
        del data
        after_fit_weights = copy.deepcopy(algorithm.learner_group.get_weights())
        rounds = []
        for index in range(3):
            audit.phase = f"round_{index + 1}"
            result = algorithm.train()
            row = {
                "round": algorithm.last_round,
                "learners": result.get("learners", {}),
            }
            rounds.append(row)
            write_json(OUTPUT / f"round-{index + 1:04d}.json", row)
            print(
                json.dumps(
                    {
                        "round": index + 1,
                        "selected_candidate": algorithm.last_round[
                            "selected_candidate"
                        ],
                        "sampled_periods": algorithm.sampled_periods,
                    }
                ),
                flush=True,
            )
        final_weights = copy.deepcopy(algorithm.learner_group.get_weights())
        audit.phase = "validation_final"
        final = evaluate(algorithm, validation_seeds)
        final_value = value_calibration(algorithm, validation_seeds)
        audit.phase = "checkpoint_replay"
        saved = algorithm.get_state()
        checkpoint = algorithm.save_to_path(str(OUTPUT / "checkpoint"))
        expected = algorithm.env_runner.collect(8)
        algorithm.restore_from_path(checkpoint)
        actual = algorithm.env_runner.collect(8)
        for key, rows in expected.policy_batches.items():
            for name, values in rows.items():
                np.testing.assert_array_equal(values, actual.policy_batches[key][name])
        algorithm.set_state(saved)
        algorithm.export_actors(OUTPUT / "actors")
        assert source_hashes("environments/sn_energy_market/src") == environment_before
        report = {
            "status": "passed",
            "formal_training": False,
            "test_loaded": False,
            "test_evaluated": False,
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "rounds": rounds,
            "ppo_sampled_periods": algorithm.sampled_periods,
            "critic_sampled_periods": sum(row["periods"] for row in critic_episodes),
            "critic_episodes": critic_episodes,
            "critic_fit": fitted,
            "initial": initial,
            "final": final,
            "value_calibration_initial": initial_value,
            "value_calibration_final": final_value,
            "critic_prefit_parameter_changes": fit_changes,
            "ppo_parameter_changes": parameter_changes(
                after_fit_weights, final_weights
            ),
            "checkpoint_exact_replay": True,
            "environment_source_unchanged": True,
            "environment_source_sha256": environment_before,
            "learning_contract": configuration.learning_contract(),
            "diagnostics": audit.report(),
            "source_sha256": {
                **source_hashes("RLAlgo/src/raymarl/algorithms/stackelberg"),
                **source_hashes("experiments/sn_energy_market"),
            },
        }
        write_json(OUTPUT / "report.json", report)
        write_json(OUTPUT / "optimizer_steps.json", audit.optimizer_steps)
        print(
            json.dumps({"status": "passed", "report": str(OUTPUT / "report.json")}),
            flush=True,
        )
    except Exception as exc:
        write_json(
            OUTPUT / "failure.json",
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
