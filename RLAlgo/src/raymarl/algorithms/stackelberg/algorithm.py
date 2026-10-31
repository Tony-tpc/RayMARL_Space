"""Native RLlib lifecycle with matched-response leader candidate selection.

algorithm.train()
    ↓
NativeAlgorithmMixin.train()       公共入口，附加日志和 checkpoint
    ↓
RLlib 的训练生命周期
    ↓
Stackelberg.training_step()        本算法真正的一整轮训练
    ↓
多次 _adapt(...)                   每次完成一批同阶段采样和学习
    ↓
learner_group.update(...)          进入 Learner 的 minibatch 更新
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.metrics import LEARNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME

from raymarl.common.algorithm import NativeAlgorithmMixin

from .batch_log import BatchLossLog
from .critic_replay import CriticReplay
from .evaluation import (
    actor_parameter_change,
    paired_gain,
    select_candidate,
    stopping_status,
    summarize_evaluation,
)
from .runner import random_state, restore_random_state


class Stackelberg(NativeAlgorithmMixin, Algorithm):
    """One train() call completes one lower/upper response-evaluation round."""

    def _setup_logdir(self):
        # Local execution writes directly to the requested output directory and
        # leaves the caller's working directory available to scenario factories.
        self._logdir = str(Path(self._storage.trial_fs_path).resolve())
        Path(self._logdir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_default_config(cls):
        from .config import StackelbergConfig

        return StackelbergConfig(algo_class=cls)

    def setup(self, config):
        super().setup(config)
        self.sampled_periods = self.evaluated_periods = self.completed_updates = self.outer_round = 0
        self.mode_sampled_periods = 0
        self.last_round = None
        self.stable_rounds = 0
        self.uc_critic_replay = CriticReplay(self.config.uc_critic_replay_batches, self.config.seed)
        self.batch_loss_log = BatchLossLog(Path(self.logdir) / "batch_losses.jsonl")

    @property
    def consumer_ids(self):
        return tuple(key for key in self.config.agent_order if key != "uc")

    def _sync(self):
        self.env_runner.module.set_state(self.learner_group.get_weights())
        self.env_runner._weights_seq_no = self.learner_group._learner._weights_seq_no

    def _snapshot(self):
        # Runner already owns an independent environment snapshot. Copying it
        # again duplicates the entire execution history for every candidate.
        return {
            "learner": copy.deepcopy(self.learner_group.get_state()),
            "runner": self.env_runner.branch_state(),
            "rng": random_state(),
            "uc_critic_replay": self.uc_critic_replay.get_state(),
        }

    def _restore(self, snapshot):
        self.learner_group.set_state(snapshot["learner"])
        self.env_runner.set_state(snapshot["runner"])
        self._sync()
        restore_random_state(snapshot["rng"])
        self.uc_critic_replay.set_state(snapshot["uc_critic_replay"])

    def _batch_diagnostics(self, batch, actor_ids=()):
        results = self.learner_group.foreach_learner(
            lambda learner: learner.batch_diagnostics(batch, actor_ids=actor_ids)
        )
        return next(iter(results)).get()  # This algorithm requires one local learner.

    def _adapt(self, module_ids, updates, *, seed=None, phase="adapt", candidate_index=None):
        if seed is not None:
            lane = next(i for i, seeds in enumerate(self.config.train_seed_groups) if seed in seeds)
            self.env_runner._switch_lane(lane)
            self.env_runner.restart(seed)
        if updates == 0:
            return
        phase_started = time.perf_counter()
        # Keep the joint policy fixed across all sampling units in this phase.
        batch = self.env_runner.collect_phase(self.config.rollout_length * updates)
        collection_seconds = time.perf_counter() - phase_started
        before = self._batch_diagnostics(batch)
        # Critic targets and behavior logp stay fixed at collection.
        # All values track this joint policy, even when their actor is frozen.
        update_started = time.perf_counter()
        critic_batch = MultiAgentBatch({
            **batch.policy_batches, "uc": self.uc_critic_replay.mix(batch["uc"]),
        }, batch.count)
        replay_view = MultiAgentBatch({"uc": critic_batch["uc"]}, batch.count)
        replay_before = self._batch_diagnostics(replay_view)["uc"]
        critic_result = self.learner_group.update(
            batch=critic_batch,
            # Count fresh experience once; replay must not bias PopArt statistics.
            popart_targets={"uc": batch["uc"]["value_targets"]},
            update_phase="critic",
            num_epochs=self.config.critic_epochs,
            minibatch_size=self.config.minibatch_size,
            shuffle_batch_per_epoch=True,
            timesteps={NUM_ENV_STEPS_SAMPLED_LIFETIME: self.sampled_periods + batch.count},
        )
        self._sync()
        selected = MultiAgentBatch({key: batch.policy_batches[key] for key in module_ids}, batch.count)
        critic_and_sync_seconds = time.perf_counter() - update_started
        gae_started = time.perf_counter()
        refreshed = self.learner_group.foreach_learner(
            lambda learner: learner.recompute_advantages(selected)
        )
        for key, advantage in next(iter(refreshed)).get().items():
            selected[key]["advantages_raw"] = advantage.copy()
            selected[key]["advantages"] = advantage
        # Normalize once over the phase batch, then freeze across Actor epochs.
        if self.config.normalize_advantages:
            for data in selected.policy_batches.values():
                advantage = data["advantages"]
                data["advantages"] = (advantage - advantage.mean()) / max(advantage.std(), 1e-8)
        actor_before = self._batch_diagnostics(selected, module_ids)
        gae_and_diagnostics_seconds = time.perf_counter() - gae_started
        update_started = time.perf_counter()
        result = self.learner_group.update(
            batch=selected,
            update_phase="actor",
            num_epochs=self.config.num_epochs,
            minibatch_size=self.config.minibatch_size,
            shuffle_batch_per_epoch=True,
            timesteps={NUM_ENV_STEPS_SAMPLED_LIFETIME: self.sampled_periods + batch.count},
        )
        actor_seconds = time.perf_counter() - update_started
        shared_metrics = {"total_loss", "grad_norm"}
        critic_metrics = shared_metrics | {
            "value_loss", "value_rmse", "value_explained_variance", "value_learning_rate",
        }
        actor_metrics = shared_metrics | {
            "policy_loss", "entropy", "clip_fraction", "approx_kl", "actor_learning_rate",
        }
        # RLlib also returns empty statistics for frozen modules. Aggregate
        # only measurements made by this update, keeping reports finite.
        current = [
            {
                key: {name: value for name, value in update[key].items() if name in measured}
                for key in keys
            }
            for updates_for_phase, keys, measured in (
                (critic_result, self.config.agent_order, critic_metrics), (result, module_ids, actor_metrics)
            )
            for update in updates_for_phase
        ]
        self.metrics.aggregate(current, key=LEARNER_RESULTS)
        self.sampled_periods += batch.count
        self.completed_updates += 1
        self._sync()
        mode_credit = None
        if "uc" in module_ids and self.config.mode_credit is not None:
            settings = self.config.mode_credit
            mode_batch, extra_periods = self.env_runner.collect_mode_returns(
                batch.count, stride=settings["stride"], horizon=settings["horizon"],
                replicates=settings["replicates"],
                seed=int(self.config.seed) * 1_000_003 + self.completed_updates * 10_000_019,
            )
            self.mode_sampled_periods += extra_periods
            updates = self.learner_group.foreach_learner(
                lambda learner: learner.update_mode_head("uc", mode_batch, target_kl=settings["target_kl"])
            )
            mode_credit = next(iter(updates)).get()
            mode_credit.update(extra_periods=extra_periods, sampled_periods_lifetime=self.mode_sampled_periods)
            self._sync()
        after = self._batch_diagnostics(batch, module_ids)
        replay_after = self._batch_diagnostics(replay_view)["uc"]
        first = batch.policy_batches["uc"]
        self.batch_loss_log.record({
            **({"mode_credit": mode_credit} if mode_credit is not None else {}),
            "outer_round": self.outer_round + 1,
            "completed_updates": self.completed_updates,
            "phase": phase, "candidate_index": candidate_index,
            "actor_ids": list(module_ids), "samples": batch.count,
            "timing_seconds": {"collection": collection_seconds, "critic_and_sync": critic_and_sync_seconds,
                               "gae_and_diagnostics": gae_and_diagnostics_seconds, "actor": actor_seconds,
                               "phase": time.perf_counter() - phase_started},
            "sampled_periods_lifetime": self.sampled_periods,
            "scenario_seeds": list(dict.fromkeys(first["scenario_seed"].tolist())),
            "first_period": int(first["period_id"][0]),
            "last_period": int(first["period_id"][-1]),
            "true_terminals": int(first["terminateds"].sum()),
            "stream_samples": {str(i): int((first["stream_id"] == i).sum())
                               for i in np.unique(first["stream_id"])},
            "segment_ends": int(first["segment_end"].sum()),
            "metrics": copy.deepcopy(self.env_runner.last_metrics),
            "gamma": self.config.gamma, "lambda_gae": self.config.lambda_gae,
            "critic_lambda": self.config.critic_lambda,
            "critic_epochs": self.config.critic_epochs, "actor_epochs": self.config.num_epochs,
            "minibatch_size": self.config.minibatch_size,
            "uc_critic_replay": {
                "pool_batches": len(self.uc_critic_replay.entries),
                "sampled_rows": batch.count,
                "rmse_before": replay_before["critic_rmse"],
                "rmse_after": replay_after["critic_rmse"],
                "fresh_batch_metrics": "modules.uc.critic_rmse_before/after",
            },
            "modules": {
                key: {
                    "critic_loss_before": before[key]["critic_loss"],
                    "critic_loss_after": after[key]["critic_loss"],
                    "critic_rmse_before": before[key]["critic_rmse"],
                    "critic_rmse_after": after[key]["critic_rmse"],
                    "value_scale": after[key]["value_scale"],
                    "policy_loss_before": actor_before[key]["policy_loss"] if key in module_ids else None,
                    "policy_loss_after": after[key]["policy_loss"],
                    "policy_loss": after[key]["policy_loss"],
                    "diagnostics_before": before[key]["value_diagnostics"],
                    "diagnostics_after": after[key]["value_diagnostics"],
                    "critic_gradient_norms": after[key]["critic_gradient_norms"],
                    "mode_diagnostics": after[key]["mode_diagnostics"],
                }
                for key in batch.policy_batches
            },
        })

    def _evaluate(self, seeds, deterministic=False, *, actions_only=True):
        rows = self.env_runner.evaluate(
            seeds, steps=self.config.evaluation_steps, deterministic=deterministic,
            actions_only=actions_only,
        )
        self.evaluated_periods += sum(row["physical_periods"] for row in rows)
        return rows

    def evaluate_pair(self, seeds=None, *, deterministic=False):
        """Frozen policy performance; default split is the held-out test set."""
        seeds = tuple(self.config.test_seeds if seeds is None else seeds)
        if not seeds:
            raise ValueError("No test scenarios configured; pass explicit nonempty evaluation seeds")
        rows = self._evaluate(seeds, deterministic, actions_only=False)
        return {"rows": rows, "summary": summarize_evaluation(rows), "deterministic": deterministic}

    def search_deviations(self):
        """Train one consumer at a time with every other policy fixed, then restore."""
        if self.config.deviation_updates == 0:
            return {"searched": False, "per_consumer": {}, "maximum_found_gain": None}
        reference = self._snapshot()
        baseline = self._evaluate(self.config.validation_seeds)
        results = {}
        seed = self.config.train_seeds[self.outer_round % len(self.config.train_seeds)]
        try:
            for key in self.consumer_ids:
                self._restore(reference)
                self._adapt((key,), self.config.deviation_updates, seed=seed, phase="deviation")
                rows = self._evaluate(self.config.validation_seeds)
                result = paired_gain(rows, baseline, agent=key, z=self.config.acceptance_z)
                results[key] = {**result, "found_gain": max(0.0, result["mean"])}
        finally:
            self._restore(reference)
        return {
            "searched": True,
            "per_consumer": results,
            "maximum_found_gain": max(item["found_gain"] for item in results.values()),
            "updates_per_consumer": self.config.deviation_updates,
            "validation_seeds": list(self.config.validation_seeds),
        }

    def _select_candidate(self, evaluated):
        selected, gains = select_candidate(evaluated, z=self.config.acceptance_z, min_gain=self.config.min_uc_gain)
        self.gate_selected = selected
        return (1 if self.config.always_keep_candidate else selected), gains

    def training_step(self):
        try:
            return self._training_step()
        finally:
            self.env_runner.release_branches()

    def _training_step(self):
        """
        训练一轮 Stackelberg 迭代。每轮迭代包括：
        1. 适应消费者策略（inner loop）。
        2. 搜索偏差（可选）。
        3. 评估候选策略。
        4. 选择候选策略。
        5. 更新日志和指标。
        """

        # 给配置起一个简短的名字 cfg。开始新一轮日志，清空上一轮尚待标记的候选日志列表。保存轮初网络权重，供最后计算“这一轮 Actor 改变了多少”。这里还没有训练。
        cfg = self.config
        self.env_runner.begin_round(self.outer_round)
        self.batch_loss_log.begin_round()
        starting_weights = copy.deepcopy(self.learner_group.get_weights())
        # Only a true episode boundary rotates the main training scenario.
        self._adapt(self.consumer_ids, # 本阶段允许更新 Actor 的智能体是全部消费者
                    cfg.inner_updates, # 做多少批“重新采样＋学习”
                    phase="consumer_adaptation" # 给日志标记这个阶段的名字
                    )
        deviations = self.search_deviations()
        reference = self._snapshot()
        candidates = [reference]
        for index in range(1, cfg.uc_candidates + 1):
            self._adapt(("uc",), cfg.uc_updates_per_candidate, phase="uc_proposal", candidate_index=index)
            candidates.append(self._snapshot())

        pairs = []
        for index, candidate in enumerate(candidates):
            self._restore(candidate)
            # Proposals now train all critics. Discard their consumer value/Adam
            # progress so every response has identical consumer starting state.
            common = reference["learner"][COMPONENT_LEARNER]
            self.learner_group.set_state({COMPONENT_LEARNER: {
                COMPONENT_RL_MODULE: {
                    key: common[COMPONENT_RL_MODULE][key] for key in self.consumer_ids
                },
                COMPONENT_OPTIMIZER: {
                    name: state for name, state in common[COMPONENT_OPTIMIZER].items()
                    if state["module_id"] in self.consumer_ids
                },
                "weights_seq_no": candidate["learner"][COMPONENT_LEARNER]["weights_seq_no"],
            }})
            self._sync()
            # Discard proposal sampling progress, retaining its UC weights/Adam.
            # Every response starts at the same physical state AND random stream.
            self.env_runner.set_state({"sampling": reference["runner"]["sampling"]})
            self._adapt(self.consumer_ids, cfg.response_updates, phase="response", candidate_index=index)
            pairs.append(self._snapshot())
        evaluated = self.env_runner.evaluate_candidates(
            [pair["runner"][COMPONENT_RL_MODULE] for pair in pairs],
            cfg.validation_seeds, steps=cfg.evaluation_steps,
        )
        self.evaluated_periods += sum(row["physical_periods"] for rows in evaluated for row in rows)
        selected, gains = self._select_candidate(evaluated)
        self._restore(pairs[selected])
        self.batch_loss_log.resolve(selected)
        final_deviations = self.search_deviations()
        policy_change = actor_parameter_change(starting_weights, self.learner_group.get_weights())
        stopping = stopping_status(final_deviations, gains, policy_change, cfg.stopping_tolerances)
        self.stable_rounds = self.stable_rounds + 1 if stopping["passed"] else 0
        self.outer_round += 1
        self.last_round = {
            "outer_round": self.outer_round,
            "selected_candidate": selected,
            "gate_selected_candidate": self.gate_selected,
            "candidate_accepted": selected != 0,
            "candidates": [
                {
                    "index": i,
                    "paired_uc_gain": gains[i],
                    "evaluation": summarize_evaluation(rows),
                    "rows": rows,
                }
                for i, rows in enumerate(evaluated)
            ],
            "consumer_deviations_before_proposal": deviations,
            "consumer_deviations_selected_pair": final_deviations,
            "policy_change": policy_change,
            "stopping": {**stopping, "stable_rounds": self.stable_rounds, "patience": cfg.stopping_patience},
            "stop_recommended": cfg.stopping_patience > 0 and self.stable_rounds >= cfg.stopping_patience,
            "maximum_found_uc_gain": max(0.0, max(item["mean"] for item in gains)),
            "budget": {
                "inner_updates": cfg.inner_updates,
                "response_updates_per_candidate": cfg.response_updates,
                "proposal_updates_per_candidate": cfg.uc_updates_per_candidate,
                "physical_periods_per_update": cfg.rollout_length,
                "ppo_epochs": cfg.num_epochs,
                "critic_epochs": cfg.critic_epochs,
                "minibatch_size": cfg.minibatch_size,
            },
            "sampled_periods_lifetime": self.sampled_periods,
            "main_periods_lifetime": self.env_runner.trajectory_periods,
            "main_scenario_seed": self.env_runner._scenario_seed,
            "evaluated_periods_lifetime": self.evaluated_periods,
            "updates_lifetime": self.completed_updates,
        }
        if self.config.mode_credit is not None:
            self.last_round["mode_sampled_periods_lifetime"] = self.mode_sampled_periods
        self.metrics.log_value(NUM_ENV_STEPS_SAMPLED_LIFETIME, self.sampled_periods, window=1)
        self.metrics.log_value("num_updates_lifetime", self.completed_updates, window=1)
        self.metrics.log_value("selected_candidate", selected, window=1)
        self.metrics.log_value(
            "response_uc_return", summarize_evaluation(evaluated[selected])["returns"]["uc"], window=1
        )
        path = Path(self.logdir) / "rounds.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.last_round, ensure_ascii=False, allow_nan=False) + "\n")

    def get_state(self, *args, **kwargs):
        state = super().get_state(*args, **kwargs)
        state["learning_contract"] = self.config.learning_contract()
        state["uc_critic_replay"] = self.uc_critic_replay.get_state()
        state["stackelberg_runtime"] = {
            "sampled_periods": self.sampled_periods,
            "evaluated_periods": self.evaluated_periods,
            "completed_updates": self.completed_updates,
            "outer_round": self.outer_round,
            "last_round": copy.deepcopy(self.last_round),
            "stable_rounds": self.stable_rounds,
        }
        if self.config.mode_credit is not None:
            state["stackelberg_runtime"]["mode_sampled_periods"] = self.mode_sampled_periods
        return state

    def set_state(self, state):
        if state["learning_contract"] != self.config.learning_contract():
            raise ValueError("Checkpoint learning contract differs from the configured model/scenarios")
        super().set_state(state)
        self.uc_critic_replay.set_state(state["uc_critic_replay"])
        self.mode_sampled_periods = state["stackelberg_runtime"]["mode_sampled_periods"] if self.config.mode_credit is not None else 0
        for key, value in state["stackelberg_runtime"].items():
            setattr(self, key, copy.deepcopy(value))
        self._sync()

    def save_to_path(self, path=None, **kwargs):
        saved = super().save_to_path(path, **kwargs)
        (Path(saved) / "learning_contract.json").write_text(
            json.dumps(self.config.learning_contract(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return saved

    def restore_from_path(self, path, *args, **kwargs):
        contract = json.loads((Path(path) / "learning_contract.json").read_text(encoding="utf-8"))
        if contract != self.config.learning_contract():
            raise ValueError("Checkpoint learning contract differs from the configured model/scenarios")
        return super().restore_from_path(path, *args, **kwargs)

    def export_actors(self, path):
        from .inference import export_actors

        return export_actors(self, path)
