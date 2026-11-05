"""Native RLlib lifecycle, joint replay orchestration and compact local outputs."""

import copy
import json
import random
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, LEARNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.train._internal.storage import StorageContext

from .connectors import CRITIC_ID
from .replay import JointReplay
from .targets import validate_finite

SCHEMA = 2


class NativeAlgorithmMixin:
    def __init__(self, config=None, **kwargs):
        cfg = config or self.get_default_config()
        if isinstance(cfg, dict):
            cfg = self.get_default_config().update_from_dict(cfg)
        cfg = cfg.copy(copy_frozen=False)
        cfg.prepare_policies()
        cfg.validate()
        if kwargs.get("storage") is None:
            kwargs["storage"] = StorageContext(
                storage_path=Path(cfg.storage_path).resolve().as_posix(),
                experiment_dir_name="raymarl",
                trial_dir_name=f"{type(self).__name__}-{uuid4().hex[:12]}",
            )
        super().__init__(config=cfg, **kwargs)

    def train(self):
        result = super().train()
        root = Path(self.logdir)
        root.mkdir(parents=True, exist_ok=True)
        env_metrics = result.get("env_runners", {})
        important = {
            "actor_loss",
            "critic_loss",
            "policy_loss",
            "value_loss",
            "value_rmse",
            "value_explained_variance",
            "total_loss",
            "actor_grad_norm",
            "critic_grad_norm",
            "num_updates",
        }
        learner_metrics = {
            mid: {
                key: float(value)
                for key, value in metrics.items()
                if (key in important or key.endswith("_learning_rate"))
                and isinstance(value, (int, float, np.number))
            }
            for mid, metrics in result.get("learners", {}).items()
        }
        summary = {
            "iteration": result["training_iteration"],
            "env_steps": result.get(NUM_ENV_STEPS_SAMPLED_LIFETIME),
            "updates": result.get("num_updates_lifetime"),
            "episode_return_mean": env_metrics.get("episode_return_mean"),
            "episode_len_mean": env_metrics.get("episode_len_mean"),
            "agent_returns": env_metrics.get("agent_episode_returns_mean", {}),
            "learners": {mid: values for mid, values in learner_metrics.items() if values},
        }
        with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")
        if self.config.detailed_outputs:
            with (root / "details.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        interval = self.config.checkpoint_interval
        if interval and result["training_iteration"] % interval == 0:
            self.save_to_path()
        return result

    def save_to_path(self, path=None, **kwargs):
        managed = path is None
        if managed:
            path = (
                Path(self.logdir)
                / "checkpoints"
                / f"checkpoint-{self.training_iteration:08d}-{uuid4().hex[:8]}"
            )
        saved = super().save_to_path(str(path), **kwargs)
        marker = Path(saved) / ".raymarl-managed"
        if managed:
            marker.write_text("raymarl-managed-v2", encoding="utf-8")
            root = (Path(self.logdir) / "checkpoints").resolve()
            paths = sorted(
                (p for p in root.glob("checkpoint-*") if (p / ".raymarl-managed").is_file()),
                key=lambda p: p.stat().st_mtime_ns,
            )
            for old in paths[: -self.config.keep_checkpoints]:
                if old.is_symlink() or old.resolve().parent != root:
                    raise ValueError("Refusing checkpoint cleanup outside the run directory")
                shutil.rmtree(old)
        elif marker.is_file() and marker.read_text(encoding="utf-8") == "raymarl-managed-v2":
            marker.unlink()
        return saved

    def get_state(self, *args, **kwargs):
        state = super().get_state(*args, **kwargs)
        state["raymarl_schema"] = SCHEMA
        state["raymarl_algorithm"] = type(self).__name__
        state["raymarl_rng"] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        return copy.deepcopy(state)

    def set_state(self, state):
        if state.get("raymarl_schema") != SCHEMA or state.get("raymarl_algorithm") != type(self).__name__:
            raise ValueError("Incompatible checkpoint: use a new-format checkpoint for this algorithm")
        validate_finite(state)
        super().set_state(copy.deepcopy(state))
        rng = state["raymarl_rng"]
        torch.set_rng_state(torch.as_tensor(rng["torch"], dtype=torch.uint8).cpu())
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([torch.as_tensor(x, dtype=torch.uint8).cpu() for x in rng["cuda"]])

    def export_actors(self, path):
        from raymarl.inference import export_actor_bundle

        return export_actor_bundle(self, path)


class OffPolicyAlgorithm(NativeAlgorithmMixin, Algorithm):
    def setup(self, config):
        super().setup(config)
        self.replay = JointReplay(self.config.replay_capacity, self.config.seed)
        self.sampled_steps = 0
        self.completed_updates = 0

    def training_step(self):
        previous = self.sampled_steps
        episodes, metrics = synchronous_parallel_sample(
            worker_set=self.env_runner_group,
            concat=True,
            sample_timeout_s=self.config.sample_timeout_s,
            _uses_new_env_runners=True,
            _return_metrics=True,
        )
        self.metrics.aggregate(metrics, key=ENV_RUNNER_RESULTS)
        for episode in episodes:
            length = episode.env_t
            if length > self.config.max_episode_steps:
                raise RuntimeError("Environment exceeded max_episode_steps")
        self.replay.add_episodes(episodes, self.config.agent_order)
        self.sampled_steps += sum(len(e) for e in episodes)
        start = max(previous + 1, self.config.warmup_steps)
        updates = max(
            0, self.sampled_steps // self.config.update_interval - (start - 1) // self.config.update_interval
        )
        env = self.env_runner.env.envs[0].unwrapped
        observation_spaces = {aid: env.get_observation_space(aid) for aid in self.config.agent_order}
        action_spaces = {aid: env.get_action_space(aid) for aid in self.config.agent_order}
        for _ in range(updates):
            replay_episodes = self.replay.sample_episodes(
                self.config.batch_size, self.config.policy_mapping_fn, observation_spaces, action_spaces
            )
            results = self.learner_group.update(
                episodes=replay_episodes,
                timesteps={NUM_ENV_STEPS_SAMPLED_LIFETIME: self.sampled_steps},
            )
            self.metrics.aggregate(results, key=LEARNER_RESULTS)
            self.completed_updates += 1
        if updates:
            self.env_runner_group.sync_weights(
                from_worker_or_learner_group=self.learner_group,
                policies=set(self.config.policies) - {CRITIC_ID},
                inference_only=True,
            )
        self.metrics.log_value(NUM_ENV_STEPS_SAMPLED_LIFETIME, self.sampled_steps, window=1)
        self.metrics.log_value("num_updates_lifetime", self.completed_updates, window=1)

    def get_state(self, *args, **kwargs):
        state = super().get_state(*args, **kwargs)
        state["raymarl_replay"] = self.replay.get_state()
        state["raymarl_counters"] = (self.sampled_steps, self.completed_updates)
        state["raymarl_layout"] = (self.config.agent_order, self.config.agent_to_module)
        return state

    def set_state(self, state):
        if state.get("raymarl_schema") != SCHEMA:
            raise ValueError("Incompatible checkpoint: old RayMARL checkpoints are not supported")
        if state.get("raymarl_layout") != (self.config.agent_order, self.config.agent_to_module):
            raise ValueError("Checkpoint agent/module mapping mismatch")
        if state.get("raymarl_replay", {}).get("capacity") != self.replay.capacity:
            raise ValueError("Checkpoint replay capacity mismatch")
        super().set_state(state)
        self.replay.set_state(state["raymarl_replay"])
        self.sampled_steps, self.completed_updates = state["raymarl_counters"]
        # Generic environments do not expose serializable simulator state.
        self.env_runner._needs_initial_reset = True
