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

    def state(self):
        return {"steps": dict(self.steps), "batches": copy.deepcopy(self.batches)}

    def load(self, state):
        self.steps = Counter(state.get("steps", {}))
        self.batches = copy.deepcopy(state.get("batches", []))

    def install(self):
        algorithm, runner = self.algorithm, self.algorithm.env_runner
        original_adapt, original_collect, original_period = (
            algorithm._adapt,
            runner.collect,
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
            assert steps == 384, "The experimental learning cut must remain 384"
            start = runner.trajectory_periods
            self.physical = []
            self.collecting = True
            try:
                batch = original_collect(steps, **kwargs)
            finally:
                self.collecting = False
            assert batch.count == 384 and runner.trajectory_periods - start == 384
            row = {
                "batch": len(self.batches) + 1,
                "round": algorithm.outer_round + 1,
                "phase": self.phase,
                "candidate": self.candidate,
                "lane": runner.active_lane,
                "samples": batch.count,
            }
            uc = batch["uc"]
            horizons, zero_tails = np.empty(384, dtype=int), np.empty(384, dtype=bool)
            length, closed = 0, False
            for t in range(383, -1, -1):
                if uc["terminateds"][t]:
                    length, closed = 0, True
                length += 1
                horizons[t], zero_tails[t] = length, closed
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
            )
            for data in batch.policy_batches.values():
                assert len(data["rewards"]) == 384
                assert all(np.isfinite(x).all() for x in data.values())
                assert (data["next_vf_preds"][data["terminateds"]] == 0).all()
                expected_tail = np.where(
                    zero_tails,
                    0,
                    algorithm.config.gamma**horizons * data["next_vf_preds"][-1],
                )
                np.testing.assert_allclose(
                    data["value_bootstrap"], expected_tail, atol=1e-9, rtol=1e-10
                )
                for t in (0, 31, 96, 383):
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
                            "lane": runner.active_lane,
                            **physical,
                            **{
                                f"prob_{name}": float(p[j])
                                for j, name in enumerate(
                                    ("idle", "charge", "discharge")
                                )
                            },
                        }
                        for physical, p in zip(self.physical, probs, strict=True)
                    ]
                    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                    if not exists:
                        writer.writeheader()
                    writer.writerows(records)
            return batch

        self.stack.enter_context(patch.object(algorithm, "_adapt", adapt))
        self.stack.enter_context(patch.object(runner, "collect", collect))
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
        assert len(self.batches) == rounds * 4
        for agent in self.algorithm.config.agent_order:
            assert self.steps[f"{agent}:value"] == rounds * 192
            assert self.steps[f"{agent}:actor"] == rounds * (
                24 if agent == "uc" else 72
            )
        assert self.algorithm.sampled_periods == rounds * 1536
        assert self.algorithm.env_runner.trajectory_periods == rounds * 768
        assert sum(self.algorithm.env_runner.lane_steps) == rounds * 768

    def close(self):
        self.stack.close()
        for handle in self.handles:
            handle.remove()
        self.algorithm.experiment_audit = None
