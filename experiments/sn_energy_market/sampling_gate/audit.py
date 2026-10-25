"""Passive executed-budget, branch isolation and target-horizon evidence."""

from __future__ import annotations

import copy
import csv
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch

from experiments.sn_energy_market.hybrid_storage_study import physical_row
from experiments.sn_energy_market.sampling_gate.runtime import digest_tree


def consumer_state(algorithm):
    learner = algorithm.learner_group._learner
    return {
        "weights": {k: learner.module[k].get_state() for k in algorithm.consumer_ids},
        "optimizers": {
            k: opt.state_dict()
            for k, opt in learner._named_optimizers.items()
            if learner._optimizer_name_to_module[k] in algorithm.consumer_ids
        },
    }


class ExperimentAudit:
    def __init__(self, algorithm, output, *, write_trajectory=True):
        self.algorithm, self.output = algorithm, output
        self.stack, self.handles = ExitStack(), []
        self.phase, self.candidate = "setup", None
        self.steps, self.batches = Counter(), []
        self.reference_consumers = self.reference_sampling = None
        self.physical = []
        self.collecting = False
        self.write_trajectory = write_trajectory
        self.load(getattr(algorithm, "restored_audit", {}))
        self.start_round = getattr(algorithm, "outer_round", 0)
        self.start_steps = self.steps.copy()
        self.start_batches = len(self.batches)
        self.start_sampled = getattr(algorithm, "sampled_periods", 0)
        self.start_physical = algorithm.env_runner.trajectory_periods
        self.start_lanes = sum(algorithm.env_runner.lane_steps)

    def state(self):
        return {"steps": dict(self.steps), "batches": copy.deepcopy(self.batches)}

    def load(self, state):
        self.steps = Counter(state.get("steps", {}))
        self.batches = copy.deepcopy(state.get("batches", []))

    def install(self):
        algorithm, runner = self.algorithm, self.algorithm.env_runner
        original_adapt, original_collect, original_period = (
            algorithm._adapt,
            runner.collect_phase,
            runner._period,
        )

        def adapt(module_ids, updates, **kwargs):
            self.phase, self.candidate = (
                kwargs.get("phase", "adapt"),
                kwargs.get("candidate_index"),
            )
            if self.phase == "response":
                assert (
                    digest_tree(consumer_state(algorithm)) == self.reference_consumers
                )
                assert (
                    digest_tree(runner.get_state(components="sampling"))
                    == self.reference_sampling
                )
            result = original_adapt(module_ids, updates, **kwargs)
            if self.phase == "consumer_adaptation":
                self.reference_consumers = digest_tree(consumer_state(algorithm))
                self.reference_sampling = digest_tree(
                    runner.get_state(components="sampling")
                )
            return result

        def period(**kwargs):
            before = runner.env.engine.state
            samples, result = original_period(**kwargs)
            if self.collecting:
                self.physical.append(physical_row(runner.env, before, samples, result))
            return samples, result

        def collect(steps, **kwargs):
            start = runner.trajectory_periods
            self.physical = []
            self.collecting = True
            try:
                batch = original_collect(steps, **kwargs)
            finally:
                self.collecting = False
            assert batch.count == steps and runner.trajectory_periods - start == steps
            row = {
                "batch": len(self.batches) + 1,
                "round": algorithm.outer_round + 1,
                "phase": self.phase,
                "candidate": self.candidate,
                "lane": runner.active_lane,
                "samples": batch.count,
            }
            uc = batch["uc"]
            horizons, zero_tails = np.empty(steps, dtype=int), np.empty(steps, dtype=bool)
            ends = np.empty(steps, dtype=int)
            length, closed, end = 0, False, steps - 1
            for t in range(steps - 1, -1, -1):
                if uc["segment_end"][t] or uc["terminateds"][t]:
                    length, closed, end = 0, bool(uc["terminateds"][t]), t
                length += 1
                horizons[t], zero_tails[t] = length, closed
                ends[t] = end
            row.update(
                horizon_counts={
                    str(k): int(v)
                    for k, v in zip(*np.unique(horizons, return_counts=True))
                },
                true_terminals=int(uc["terminateds"].sum()),
                zero_tail_samples=int(zero_tails.sum()),
                scenario_seeds=list(dict.fromkeys(map(int, uc["scenario_seed"]))),
                first_period=int(uc["period_id"][0]),
                last_period=int(uc["period_id"][-1]),
                stream_samples={str(k): int(v) for k, v in zip(*np.unique(uc["stream_id"], return_counts=True))},
                segment_ends=int(uc["segment_end"].sum()),
            )
            for data in batch.policy_batches.values():
                assert len(data["rewards"]) == steps
                assert all(np.isfinite(x).all() for x in data.values())
                assert (data["next_vf_preds"][data["terminateds"]] == 0).all()
                expected_tail = np.where(
                    zero_tails,
                    0,
                    algorithm.config.gamma**horizons * data["next_vf_preds"][ends],
                )
                np.testing.assert_allclose(
                    data["value_bootstrap"], expected_tail, atol=1e-9, rtol=1e-10
                )
                for t in sorted({0, steps // 2, steps - 1, *np.flatnonzero(uc["segment_end"])}):
                    h = horizons[t]
                    realized = np.dot(
                        algorithm.config.gamma ** np.arange(h),
                        data["rewards"][t : t + h],
                    )
                    np.testing.assert_allclose(
                        data["value_targets"][t],
                        realized + expected_tail[t],
                        atol=1e-8,
                        rtol=1e-10,
                    )
            self.batches.append(row)
            row["batch_sha256"] = digest_tree(batch.policy_batches)
            if self.write_trajectory:
                with torch.no_grad():
                    values = {k: torch.as_tensor(v) for k, v in uc.items()}
                    probs = (
                        runner.module["uc"]
                        .evaluate_actions(values)["mode_probs"]
                        .cpu()
                        .numpy()
                    )
                target = self.output / "training-all-branches.csv"
                exists = target.exists()
                with target.open("a", encoding="utf-8", newline="") as handle:
                    records = [
                        {
                            "batch": row["batch"],
                            "round": row["round"],
                            "phase": self.phase,
                            "candidate": self.candidate,
                            "lane": int(lane),
                            **physical,
                            **{
                                f"prob_{name}": float(p[j])
                                for j, name in enumerate(
                                    ("idle", "charge", "discharge")
                                )
                            },
                        }
                        for physical, p, lane in zip(self.physical, probs, uc["stream_id"], strict=True)
                    ]
                    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                    if not exists:
                        writer.writeheader()
                    writer.writerows(records)
            return batch

        self.stack.enter_context(patch.object(algorithm, "_adapt", adapt))
        self.stack.enter_context(patch.object(runner, "collect_phase", collect))
        self.stack.enter_context(patch.object(runner, "_period", period))
        learner = algorithm.learner_group._learner
        for name, optimizer in learner._named_optimizers.items():
            agent = learner._optimizer_name_to_module[name]
            params = [p for group in optimizer.param_groups for p in group["params"]]
            actor = {id(p) for p in learner.module[agent].actor.parameters()}
            role = "actor" if id(params[0]) in actor else "value"

            def after(opt, args, kwargs, *, params=params, agent=agent, role=role):
                if any(p.grad is not None for p in params):
                    self.steps[f"{agent}:{role}"] += 1

            self.handles.append(optimizer.register_step_post_hook(after))
        algorithm.experiment_audit = self
        return self

    def assert_budget(self, rounds):
        cfg = self.algorithm.config
        elapsed = rounds - self.start_round
        sizes = [cfg.rollout_length * units for units in (
            cfg.inner_updates, *([cfg.uc_updates_per_candidate] * cfg.uc_candidates),
            *([cfg.response_updates] * (cfg.uc_candidates + 1)),
        )]
        mini = lambda size: (size + cfg.minibatch_size - 1) // cfg.minibatch_size
        assert len(self.batches) - self.start_batches == elapsed * len(sizes)
        for agent in self.algorithm.config.agent_order:
            assert self.steps[f"{agent}:value"] - self.start_steps[f"{agent}:value"] == elapsed * cfg.critic_epochs * sum(map(mini, sizes))
            actor_batches = sizes[1:1 + cfg.uc_candidates] if agent == "uc" else [sizes[0], *sizes[1 + cfg.uc_candidates:]]
            assert self.steps[f"{agent}:actor"] - self.start_steps[f"{agent}:actor"] == elapsed * cfg.num_epochs * sum(map(mini, actor_batches))
        physical = elapsed * cfg.rollout_length * (cfg.inner_updates + cfg.response_updates)
        assert self.algorithm.sampled_periods - self.start_sampled == elapsed * sum(sizes)
        assert self.algorithm.env_runner.trajectory_periods - self.start_physical == physical
        assert sum(self.algorithm.env_runner.lane_steps) - self.start_lanes == physical

    def close(self):
        self.stack.close()
        for handle in self.handles:
            handle.remove()
        self.algorithm.experiment_audit = None
