"""Native RLlib configuration with explicit model and output ownership."""

import math
import warnings
from pathlib import Path

import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.connectors.learner import AddNextObservationsFromEpisodesToTrainBatch
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.env_context import EnvContext
from ray.rllib.policy.policy import PolicySpec

from .connectors import CRITIC_ID, JointBatch, JointRowIdentity
from .multi_module import TrainingMultiRLModule


class IdentityMapping:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, agent_id, *args, **kwargs):
        return self.mapping[agent_id]


class ModelConfigMixin:
    def _init_outputs(self):
        self.storage_path = str(Path.cwd() / "outputs")
        self.keep_checkpoints = 2
        self.checkpoint_interval = 10
        self.detailed_outputs = False
        self._legacy_hidden_dim = None

    def training(self, **kwargs):
        if "hidden_dim" in kwargs:
            hidden = kwargs.pop("hidden_dim")
            if (
                self._model_config
                and "hidden_dim" in self._model_config
                and self._model_config["hidden_dim"] != hidden
            ):
                raise ValueError("Conflicting training(hidden_dim) and rl_module(model_config)")
            warnings.warn(
                "Use .rl_module(model_config={'hidden_dim': ...})", DeprecationWarning, stacklevel=2
            )
            self._legacy_hidden_dim = hidden
            self._model_config = {**(self._model_config or {}), "hidden_dim": hidden}
        for key in ("storage_path", "keep_checkpoints", "checkpoint_interval", "detailed_outputs"):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        return super().training(**kwargs)

    def rl_module(self, *, model_config=None, **kwargs):
        if (
            model_config is not None
            and self._legacy_hidden_dim is not None
            and model_config.get("hidden_dim", self._legacy_hidden_dim) != self._legacy_hidden_dim
        ):
            raise ValueError("Conflicting training(hidden_dim) and rl_module(model_config)")
        return super().rl_module(model_config=model_config, **kwargs)

    def validate(self):
        super().validate()
        hidden = self.model_config.get("hidden_dim", 64)
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(self.keep_checkpoints, int) or self.keep_checkpoints < 1:
            raise ValueError("keep_checkpoints must be positive")
        if not isinstance(self.checkpoint_interval, int) or self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be a non-negative integer")

    def build_algo(self, env=None, use_copy=True):
        if env is not None:
            config = self.copy(copy_frozen=False) if use_copy else self
            config.environment(env)
            return super(ModelConfigMixin, config).build_algo(use_copy=False)
        return super().build_algo(use_copy=use_copy)

    def prepare_policies(self):
        _, creator = Algorithm._get_env_id_and_creator(self.env, self)
        env = creator(EnvContext(self.env_config, worker_index=0))
        try:
            agents = tuple(sorted(env.possible_agents, key=str))
            if not agents:
                raise ValueError("A fixed nonempty possible_agents collection is required")
            if set(self.policies) == {"default_policy"}:
                mapping = {aid: str(aid) for aid in agents}
                if len(set(mapping.values())) != len(agents) or CRITIC_ID in mapping.values():
                    mapping = {aid: f"agent_{i}" for i, aid in enumerate(agents)}
                self.multi_agent(
                    policies={
                        mid: PolicySpec(
                            observation_space=env.get_observation_space(aid),
                            action_space=env.get_action_space(aid),
                        )
                        for aid, mid in mapping.items()
                    },
                    policy_mapping_fn=IdentityMapping(mapping),
                )
            else:
                mapping = {aid: self.policy_mapping_fn(aid, None) for aid in agents}
                if len(set(mapping.values())) != len(agents):
                    raise ValueError("Actor export and joint training require one policy per agent")
                if set(mapping.values()) != set(self.policies) or CRITIC_ID in self.policies:
                    raise ValueError("Policies must match the fixed agent mapping")
            self.agent_order = agents
            self.agent_to_module = mapping
        finally:
            env.close()


class OffPolicyConfig(ModelConfigMixin, AlgorithmConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class)
        self._init_outputs()
        self.actor_lr, self.critic_lr = 3e-4, 1e-3
        self.temperature, self.tau, self.gamma = 0.1, 0.001, 0.99
        self.batch_size, self.replay_capacity, self.warmup_steps = 128, 100_000, 512
        self.update_interval = 4
        self.actor_grad_clip, self.critic_grad_clip = 0.5, 50.0
        self.max_episode_steps = 100_000
        self.exploration_std = 0.1
        self.seed = 0
        self.framework_str = "torch"
        self.num_env_runners = self.num_learners = self.num_gpus_per_learner = 0
        self.rollout_fragment_length = 4
        self.normalize_actions = self.clip_actions = False
        self.min_time_s_per_iteration = 0
        self.min_sample_timesteps_per_iteration = 0
        self.evaluation_duration = 1
        self.evaluation_duration_unit = "episodes"
        self.evaluation_interval = None

    def training(self, **kwargs):
        for key in (
            "actor_lr",
            "critic_lr",
            "temperature",
            "tau",
            "batch_size",
            "replay_capacity",
            "warmup_steps",
            "update_interval",
            "actor_grad_clip",
            "critic_grad_clip",
            "max_episode_steps",
            "exploration_std",
        ):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        return super().training(**kwargs)

    def validate(self):
        super().validate()
        if not self.enable_rl_module_and_learner or not self.enable_env_runner_and_connector_v2:
            raise ValueError("Only the new RLlib API stack is supported")
        if self.num_env_runners != 0 or self.num_learners != 0:
            raise ValueError("This release supports one local EnvRunner and local Learner")
        if self.num_gpus_per_learner not in (0, 1) or self.framework_str != "torch":
            raise ValueError("Use PyTorch with CPU or a single local GPU")
        for key in ("actor_lr", "critic_lr", "tau", "actor_grad_clip", "critic_grad_clip"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.tau > 1 or not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid tau/gamma")
        for key in ("temperature", "exploration_std"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and non-negative")
        for key in ("batch_size", "replay_capacity", "warmup_steps", "update_interval", "max_episode_steps"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")

    def build_learner_connector(self, input_observation_space, input_action_space, device=None):
        pipeline = super().build_learner_connector(input_observation_space, input_action_space, device)
        pipeline.insert_before("AddObservationsFromEpisodesToBatch", JointRowIdentity())
        pipeline.insert_after(
            "AddObservationsFromEpisodesToBatch", AddNextObservationsFromEpisodesToTrainBatch()
        )
        pipeline.append(JointBatch())
        return pipeline


class JointConfig(OffPolicyConfig):
    def get_multi_rl_module_spec(self, **kwargs):
        spec = super().get_multi_rl_module_spec(**kwargs)
        spec.multi_rl_module_class = TrainingMultiRLModule
        if not kwargs.get("inference_only", False):
            ids = tuple(sorted(spec.rl_module_specs))
            specs = spec.rl_module_specs
            spec.add_modules(
                {
                    CRITIC_ID: RLModuleSpec(
                        module_class=self.critic_module_class,
                        learner_only=True,
                        observation_space=gym.spaces.Dict({mid: specs[mid].observation_space for mid in ids}),
                        action_space=gym.spaces.Dict({mid: specs[mid].action_space for mid in ids}),
                        model_config={**self.model_config, "actor_ids": ids},
                    )
                }
            )
        return spec
