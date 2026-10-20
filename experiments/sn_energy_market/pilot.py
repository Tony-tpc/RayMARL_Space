"""Small real-data training audit using train and validation splits only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import secrets
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "RLAlgo/src"),
    str(ROOT / "environments/sn_energy_market/src"),
]

from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)
from experiments.sn_energy_market.train import write_json
from experiments.sn_energy_market.validation_helpers import check_record, compare_oracle


def parameter_changes(before, after):
    """Separate actor and value changes without depending on network architecture."""
    return {
        agent: {
            role: float(
                np.sqrt(
                    sum(
                        np.square(
                            np.asarray(after[agent][name], dtype=np.float64)
                            - np.asarray(value, dtype=np.float64)
                        ).sum()
                        for name, value in weights.items()
                        if name.startswith(role + ".")
                    )
                )
            )
            for role in ("actor", "value")
        }
        for agent, weights in before.items()
    }


def reference_advantages(rows, gamma, lambda_gae):
    """Explicit discounted TD sums, independent of the production recurrence."""
    deltas = (
        rows["rewards"]
        + gamma * (~rows["terminateds"]) * rows["next_vf_preds"]
        - rows["vf_preds"]
    )
    result = np.zeros(len(deltas), dtype=np.float64)
    for start in range(len(deltas)):
        factor = 1.0
        for stop in range(start, len(deltas)):
            result[start] += factor * deltas[stop]
            if rows["terminateds"][stop]:
                break
            factor *= gamma * lambda_gae
    return result


def assert_same_state(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same_state(a, b)
    else:
        np.testing.assert_array_equal(left, right)


class TrainingAudit:
    """Observe the existing collector and optimizers; leave update rules unchanged."""

    def __init__(self, algorithm, output):
        self.algorithm, self.output = algorithm, output
        self.phase = "setup"
        self.coverage, self.maximums = Counter(), {}
        self.samples, self.updates, self.optimizer_steps = [], [], []
        self.sampled_dates = defaultdict(set)
        self.reward_stats = defaultdict(
            lambda: {"count": 0, "sum": 0.0, "min": math.inf, "max": -math.inf}
        )
        self.handles, self.stack = [], ExitStack()

    def maximum(self, name, value):
        self.maximums[name] = max(self.maximums.get(name, 0.0), float(value))

    def install(self):
        runner, group = self.algorithm.env_runner, self.algorithm.learner_group
        original_period, original_collect, original_update = (
            runner._period,
            runner.collect,
            group.update,
        )

        def period(**kwargs):
            before = runner.env.engine.state
            samples, result = original_period(**kwargs)
            record = runner.env.engine.latest_record
            if (
                self.phase.startswith(("round_", "critic_collect"))
                and runner._scenario_seed in runner.config.train_seeds
            ):
                self.sampled_dates[self.phase].add(record.timestamp.isoformat())
            check_record(
                runner.env.engine,
                before,
                record.uc_action,
                record.consumer_actions,
                record,
                self.coverage,
                self.maximums,
            )
            for key, sample in samples.items():
                np.testing.assert_array_equal(
                    sample["actions"],
                    np.array(
                        list(
                            vars(
                                record.uc_action
                                if key == "uc"
                                else record.consumer_actions[key]
                            ).values()
                        )
                    ),
                )
                assert sample["rewards"] == record.rewards[key]
                stats = self.reward_stats[f"{self.phase}/{key}"]
                reward = result.rewards[key]
                assert math.isfinite(reward)
                stats["count"] += 1
                stats["sum"] += reward
                stats["min"], stats["max"] = (
                    min(stats["min"], reward),
                    max(stats["max"], reward),
                )
            if before.period_id % 48 == 24:
                for name, value in compare_oracle(
                    runner.env.engine, before, record
                ).items():
                    self.maximum("oracle_" + name, value)
                self.coverage["oracle_comparisons"] += 1
            self.coverage[f"periods/{self.phase}"] += 1
            return samples, result

        def collect(steps, **kwargs):
            import torch

            batch = original_collect(steps, **kwargs)
            assert batch.count == steps
            seeds = set()
            for key, rows in batch.policy_batches.items():
                seeds.update(int(seed) for seed in rows["scenario_seed"])
                for name, values in rows.items():
                    assert np.isfinite(values).all(), (key, name)
                expected = reference_advantages(
                    rows, runner.config.gamma, runner.config.lambda_gae
                )
                np.testing.assert_allclose(
                    rows["advantages"], expected, atol=1e-9, rtol=1e-10
                )
                np.testing.assert_allclose(
                    rows["value_targets"],
                    reference_advantages(rows, runner.config.gamma, runner.config.critic_lambda)
                    + rows["vf_preds"],
                    atol=1e-9,
                    rtol=1e-10,
                )
                self.maximum(
                    "gae_residual", np.max(np.abs(rows["advantages"] - expected))
                )
                self.maximum(
                    "value_target_residual",
                    np.max(np.abs(rows["value_targets"] - rows["vf_preds"]
                                  - reference_advantages(
                                      rows, runner.config.gamma, runner.config.critic_lambda
                                  ))),
                )
                with torch.no_grad():
                    tensors = {
                        name: torch.as_tensor(values) for name, values in rows.items()
                    }
                    replay = (
                        runner.module[key]
                        .evaluate_actions(tensors)["action_logp"]
                        .cpu()
                        .numpy()
                    )
                np.testing.assert_allclose(
                    replay, rows["action_logp"], atol=3e-5, rtol=1e-6
                )
                self.maximum(
                    "behavior_logp_residual",
                    np.max(np.abs(replay - rows["action_logp"])),
                )
                self.coverage["zero_width_coordinates"] += int(
                    rows["zero_width_mask"].sum()
                )
                self.coverage["sampled_agent_rows"] += len(rows["rewards"])
                assert rows["truncateds"][-1] == (not rows["terminateds"][-1])
                assert (rows["next_vf_preds"][rows["terminateds"]] == 0).all()
            assert seeds <= set(runner.config.train_seeds), seeds
            self.samples.append(
                {
                    "phase": self.phase,
                    "periods": batch.count,
                    "seeds": sorted(seeds),
                    "terminated_rows": int(
                        batch.policy_batches["uc"]["terminateds"].sum()
                    ),
                    "metrics": dict(runner.last_metrics),
                }
            )
            return batch

        def update(*args, **kwargs):
            batch = kwargs["batch"]
            selected = set(batch.policy_batches)
            update_phase = kwargs.get("update_phase", "joint")
            if update_phase != "critic":
                for rows in batch.policy_batches.values():
                    advantage = rows["advantages"]
                    assert abs(advantage.mean()) < 1e-8
                    if advantage.std() > 1e-8:
                        assert abs(advantage.std() - 1.0) < 1e-8
            if update_phase == "critic":
                assert selected == set(runner.agent_ids)
            before = copy.deepcopy(group.get_weights())
            for key in selected:
                for name, value in before[key].items():
                    np.testing.assert_array_equal(
                        value,
                        runner.module[key].state_dict()[name].detach().cpu().numpy(),
                    )
            result = original_update(*args, **kwargs)
            changes = parameter_changes(before, group.get_weights())
            for key, roles in changes.items():
                assert all(math.isfinite(value) for value in roles.values())
                if key not in selected:
                    assert all(value == 0.0 for value in roles.values()), (key, roles)
                if update_phase == "critic":
                    assert roles["actor"] == 0.0, (key, roles)
                if update_phase == "actor":
                    assert roles["value"] == 0.0, (key, roles)
            self.updates.append(
                {
                    "phase": self.phase,
                    "update_phase": update_phase,
                    "modules": sorted(selected),
                    "changes": changes,
                    "frozen_unchanged": True,
                    "behavior_weights_synchronized": True,
                }
            )
            return result

        self.stack.enter_context(patch.object(runner, "_period", period))
        self.stack.enter_context(patch.object(runner, "collect", collect))
        self.stack.enter_context(patch.object(group, "update", update))
        for result in group.foreach_learner(self.install_optimizer_hooks):
            result.get()
        return self

    def install_optimizer_hooks(self, learner):
        import torch

        for name, optimizer in learner._named_optimizers.items():
            agent = learner._optimizer_name_to_module[name]
            parameters = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            actor_ids = {id(p) for p in learner.module[agent].actor.parameters()}
            role = "actor" if id(parameters[0]) in actor_ids else "value"
            raw, state = {}, {}

            def gradient_hook(grad, *, index, raw):
                assert torch.isfinite(grad).all()
                raw[index] = float(grad.detach().double().square().sum())

            for index, parameter in enumerate(parameters):
                self.handles.append(
                    parameter.register_hook(
                        lambda grad, i=index, r=raw: gradient_hook(grad, index=i, raw=r)
                    )
                )

            def before_step(
                opt, args, kwargs, *, params=parameters, raw=raw, state=state
            ):
                state["parameters"] = [p.detach().clone() for p in params]
                state["active"] = any(p.grad is not None for p in params)
                state["raw_norm"] = math.sqrt(
                    sum(
                        raw.get(i, 0.0)
                        for i, p in enumerate(params)
                        if p.grad is not None
                    )
                )
                state["clipped_norm"] = math.sqrt(
                    sum(
                        float(p.grad.detach().double().square().sum())
                        for p in params
                        if p.grad is not None
                    )
                )
                # Native clipping reduces float32 tensors; this audit recomputes
                # their norm in float64, allowing the reduction rounding error.
                self.maximum("post_clip_gradient_norm", state["clipped_norm"])
                assert state["clipped_norm"] <= learner.config.grad_clip + 1e-5, (
                    state["clipped_norm"],
                    learner.config.grad_clip,
                )
                if not state["active"]:
                    state["frozen_optimizer"] = copy.deepcopy(opt.state_dict())

            def after_step(
                opt,
                args,
                kwargs,
                *,
                params=parameters,
                raw=raw,
                state=state,
                agent=agent,
                role=role,
            ):
                delta = math.sqrt(
                    sum(
                        float((p.detach().double() - old.double()).square().sum())
                        for p, old in zip(params, state.pop("parameters"), strict=True)
                    )
                )
                assert all(torch.isfinite(p).all() for p in params)
                if not state["active"]:
                    assert delta == 0.0
                    assert_same_state(state.pop("frozen_optimizer"), opt.state_dict())
                    raw.clear()
                    self.coverage["frozen_optimizer_checks"] += 1
                    return
                row = {
                    "phase": self.phase,
                    "agent": agent,
                    "role": role,
                    "raw_grad_norm": state["raw_norm"],
                    "clipped_grad_norm": state["clipped_norm"],
                    "parameter_delta": delta,
                }
                assert all(
                    math.isfinite(row[key])
                    for key in ("raw_grad_norm", "clipped_grad_norm", "parameter_delta")
                )
                self.optimizer_steps.append(row)
                raw.clear()

            self.handles.extend(
                (
                    optimizer.register_step_pre_hook(before_step),
                    optimizer.register_step_post_hook(after_step),
                )
            )

    def close(self):
        self.stack.close()
        for handle in self.handles:
            handle.remove()

    def report(self):
        gradient_summary = {}
        for key in sorted(
            {(row["agent"], row["role"]) for row in self.optimizer_steps}
        ):
            rows = [
                row
                for row in self.optimizer_steps
                if (row["agent"], row["role"]) == key
            ]
            gradient_summary["/".join(key)] = {
                "optimizer_steps": len(rows),
                "nonzero_gradient_steps": sum(row["raw_grad_norm"] > 0 for row in rows),
                "nonzero_parameter_steps": sum(
                    row["parameter_delta"] > 0 for row in rows
                ),
                "raw_norm_min": min(row["raw_grad_norm"] for row in rows),
                "raw_norm_max": max(row["raw_grad_norm"] for row in rows),
                "clipped_norm_max": max(row["clipped_grad_norm"] for row in rows),
                "parameter_delta_max": max(row["parameter_delta"] for row in rows),
            }
        return {
            "coverage": dict(self.coverage),
            "maximums": self.maximums,
            "samples": self.samples,
            "updates": self.updates,
            "gradients": gradient_summary,
            "reward_statistics": dict(self.reward_stats),
            "sampled_dates": {
                phase: sorted(dates) for phase, dates in self.sampled_dates.items()
            },
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets/ausgrid/processed/manifest.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ausgrid-pilot")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 3:
        parser.error("This diagnostic run supports one to three outer rounds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = ProfileDataset(args.dataset, splits=("train", "validation"))
    factory = DatasetEnvFactory(dataset, dataset.environment_config(), seed=42)
    train_seeds = tuple(factory.seed_splits["train"][i] for i in (0, 3, 6))
    validation_seeds = tuple(factory.seed_splits["validation"][i] for i in (0, 3, 6, 9))
    temporary = ROOT / ".tmp/ausgrid-pilot"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(temporary)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    token = ROOT / ".ray/auth_token"
    token.parent.mkdir(exist_ok=True)
    if not token.exists():
        token.write_text(secrets.token_hex(32), encoding="utf-8")
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(token)

    import ray
    import torch

    from raymarl.algorithms.stackelberg import StackelbergConfig

    torch.set_num_threads(1)
    configuration = (
        StackelbergConfig()
        .scenarios(
            factory,
            train_seeds=train_seeds,
            validation_seeds=validation_seeds,
            test_seeds=(),
        )
        .debugging(seed=42)
        .rl_module(model_config={"hidden_dim": 32})
        .training(
            gamma=0.9801,
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
    write_json(
        output / "experiment.json",
        {
            "learning_contract": configuration.learning_contract(),
            "purpose": "small_training_validation",
            "test_loaded": False,
            "test_evaluated": False,
        },
    )
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(temporary / "ray"),
    )
    algorithm, audit = None, None
    try:
        algorithm = configuration.build_algo()
        audit = TrainingAudit(algorithm, output).install()
        initial = algorithm._snapshot()
        starting_weights = copy.deepcopy(algorithm.learner_group.get_weights())
        audit.phase = "isolation_consumer"
        algorithm._adapt(algorithm.consumer_ids, 1, seed=train_seeds[0])
        consumer_probe = parameter_changes(
            starting_weights, algorithm.learner_group.get_weights()
        )
        assert consumer_probe["uc"] == {"actor": 0.0, "value": 0.0}
        assert all(
            consumer_probe[key][role] > 0
            for key in algorithm.consumer_ids
            for role in ("actor", "value")
        )
        algorithm._restore(initial)
        audit.phase = "isolation_uc"
        algorithm._adapt(("uc",), 1, seed=train_seeds[0])
        uc_probe = parameter_changes(
            starting_weights, algorithm.learner_group.get_weights()
        )
        assert uc_probe["uc"]["actor"] > 0 and uc_probe["uc"]["value"] > 0
        assert all(
            uc_probe[key] == {"actor": 0.0, "value": 0.0}
            for key in algorithm.consumer_ids
        )
        algorithm._restore(initial)
        audit.phase = "validation_before"
        baseline = algorithm.evaluate_pair(validation_seeds)
        assert all(
            all(value == 0 for value in roles.values())
            for roles in parameter_changes(
                starting_weights, algorithm.learner_group.get_weights()
            ).values()
        )
        rounds = []
        for index in range(args.rounds):
            audit.phase = f"round_{index + 1}"
            result = algorithm.train()
            row = {
                "round": algorithm.last_round,
                "learners": result.get("learners", {}),
            }
            write_json(output / f"round-{index + 1:04d}.json", row)
            rounds.append(row)
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
        audit.phase = "validation_after"
        final = algorithm.evaluate_pair(validation_seeds)
        assert all(
            all(value == 0 for value in roles.values())
            for roles in parameter_changes(
                final_weights, algorithm.learner_group.get_weights()
            ).values()
        )
        audit.phase = "checkpoint_replay"
        snapshot = algorithm.get_state()
        checkpoint = algorithm.save_to_path(str(output / "checkpoint"))
        expected = algorithm.env_runner.collect(8)
        algorithm.restore_from_path(checkpoint)
        actual = algorithm.env_runner.collect(8)
        for key, rows in expected.policy_batches.items():
            for name, values in rows.items():
                np.testing.assert_array_equal(values, actual.policy_batches[key][name])
        algorithm.set_state(snapshot)
        algorithm.export_actors(output / "actors")
        report = {
            "status": "passed",
            "formal_training": False,
            "test_loaded": False,
            "test_evaluated": False,
            "rounds": args.rounds,
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "optimization_sampled_periods": algorithm.sampled_periods,
            "optimization_updates": algorithm.completed_updates,
            "probe_periods": 128,
            "main_training_periods": algorithm.sampled_periods - 128,
            "validation_before": baseline,
            "validation_after": final,
            "parameter_changes": parameter_changes(starting_weights, final_weights),
            "isolation_probes": {"consumer_only": consumer_probe, "uc_only": uc_probe},
            "checkpoint_exact_replay": True,
            "checkpoint": str(checkpoint),
            "diagnostics": audit.report(),
            "learning_contract": configuration.learning_contract(),
        }
        paths = list((ROOT / "experiments/sn_energy_market").glob("*.py"))
        paths += list((ROOT / "RLAlgo/src/raymarl/algorithms/stackelberg").glob("*.py"))
        report["source_sha256"] = {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        }
        write_json(output / "report.json", report)
        write_json(output / "optimizer_steps.json", audit.optimizer_steps)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "main_training_periods": report["main_training_periods"],
                    "test_evaluated": False,
                    "report": str(output / "report.json"),
                }
            ),
            flush=True,
        )
        return report
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
