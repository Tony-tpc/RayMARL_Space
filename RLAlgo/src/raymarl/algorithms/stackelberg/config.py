"""Configuration for local, two-stage Stackelberg PPO."""

import copy
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec

from .contracts import ScenarioFactory


def identity_policy(agent_id, *args, **kwargs):
    return agent_id


class StackelbergConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        from .algorithm import Stackelberg
        from .runner import StackelbergEnvRunner

        super().__init__(algo_class=algo_class or Stackelberg)
        self.env_runner_cls = StackelbergEnvRunner
        self.scenario_factory = None
        self.train_seeds = tuple(range(32))
        self.validation_seeds = tuple(range(1000, 1016))
        self.test_seeds = tuple(range(2000, 2032))
        self.gamma, self.lambda_gae = 0.99, 0.95
        self.critic_lambda = 1.0
        self.actor_lr, self.critic_lr = 3e-4, 1e-3
        self.clip_param, self.vf_loss_coeff, self.entropy_coeff = 0.2, 0.5, 0.0
        self.grad_clip = 0.5
        self.rollout_length, self.num_epochs, self.minibatch_size = 128, 4, 64
        self.critic_epochs = 4
        self.inner_updates, self.response_updates = 5, 5
        self.uc_candidates, self.uc_updates_per_candidate, self.deviation_updates = 2, 1, 2
        self.acceptance_z, self.min_uc_gain = 1.96, 0.0
        self.stopping_patience = 3
        self.stopping_tolerances = {
            "consumer_gain": 0.1,
            "uc_gain": 0.1,
            "policy_change": 1e-3,
            "confidence_half_width": 0.1,
        }
        self.evaluation_steps = None
        self.normalize_advantages = True
        self.framework_str, self.seed = "torch", 0
        self.num_env_runners = self.num_learners = self.num_gpus_per_learner = 0
        self.normalize_actions = self.clip_actions = False
        self.min_time_s_per_iteration = self.min_sample_timesteps_per_iteration = 0
        self.evaluation_interval = None
        self.storage_path = str(Path.cwd() / "outputs" / "snmarket-training")
        self.keep_checkpoints, self.checkpoint_interval = 2, 1
        self.detailed_outputs = False

    def scenarios(self, factory: ScenarioFactory, *, train_seeds, validation_seeds, test_seeds):
        self.scenario_factory = factory
        self.train_seeds = tuple(train_seeds)
        self.validation_seeds = tuple(validation_seeds)
        self.test_seeds = tuple(test_seeds)
        return self

    def training(self, **kwargs):
        fields = (
            "actor_lr",
            "critic_lr",
            "critic_epochs",
            "lambda_gae",
            "critic_lambda",
            "clip_param",
            "vf_loss_coeff",
            "entropy_coeff",
            "rollout_length",
            "inner_updates",
            "response_updates",
            "uc_candidates",
            "uc_updates_per_candidate",
            "deviation_updates",
            "acceptance_z",
            "min_uc_gain",
            "stopping_patience",
            "stopping_tolerances",
            "evaluation_steps",
            "normalize_advantages",
            "storage_path",
            "keep_checkpoints",
            "checkpoint_interval",
            "detailed_outputs",
        )
        for key in fields:
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        return super().training(**kwargs)

    def prepare_policies(self):
        from .rl_module import StackelbergRLModule

        env = self.scenario_factory(self.train_seeds[0])
        self.agent_order = ("uc", *env.consumer_ids)
        self.agent_to_module = {key: key for key in self.agent_order}
        specs, policies = {}, {}
        for key in self.agent_order:
            leader = key == "uc"
            obs_dim = env.uc_observation_dim if leader else env.consumer_observation_dim
            obs = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
            action = gym.spaces.Box(-np.inf, np.inf, (6 if leader else 4,), np.float64)
            policies[key] = PolicySpec(observation_space=obs, action_space=action)
            specs[key] = RLModuleSpec(
                module_class=StackelbergRLModule,
                observation_space=obs,
                action_space=action,
                model_config={
                    **self.model_config,
                    "role": "uc" if leader else "consumer",
                    "critic_dim": obs_dim if leader else env.consumer_value_dim,
                },
            )
        self._stackelberg_spec = MultiRLModuleSpec(rl_module_specs=specs)
        self.multi_agent(policies=policies, policy_mapping_fn=identity_policy)

    def get_multi_rl_module_spec(self, **kwargs):
        return copy.deepcopy(self._stackelberg_spec)

    def get_default_rl_module_spec(self):
        from .rl_module import StackelbergRLModule

        return RLModuleSpec(module_class=StackelbergRLModule)

    def get_default_learner_class(self):
        from .learner import StackelbergTorchLearner

        return StackelbergTorchLearner

    def learning_contract(self):
        fields = (
            "gamma",
            "lambda_gae",
            "critic_lambda",
            "actor_lr",
            "critic_lr",
            "clip_param",
            "vf_loss_coeff",
            "entropy_coeff",
            "grad_clip",
            "normalize_advantages",
            "rollout_length",
            "num_epochs",
            "critic_epochs",
            "minibatch_size",
            "inner_updates",
            "response_updates",
            "uc_candidates",
            "uc_updates_per_candidate",
            "deviation_updates",
            "acceptance_z",
            "min_uc_gain",
            "stopping_patience",
            "stopping_tolerances",
            "evaluation_steps",
            "seed",
            "train_seeds",
            "validation_seeds",
            "test_seeds",
        )
        contract = {key: getattr(self, key) for key in fields}
        contract.update(
            model=self.model_config,
            scenario=self.scenario_factory.get_contract(),
            input_transform="signed_log1p",
            uc_probability="masked_categorical_conditional_active_gaussian",
            consumer_probability="conditional_beta_physical_density",
            evaluation_return="complete_finite_episode_with_terminal_inventory",
            sampling_protocol="continuous_main_common_response_snapshot_selected_branch_v1",
            update_protocol="all_critics_then_selected_actors_common_consumer_state_v1",
            value_architecture="layernorm_tanh",
        )
        return json.loads(json.dumps(contract, sort_keys=True))

    def validate(self):
        super().validate()
        if self.scenario_factory is None:
            raise ValueError("Configure a scenario factory and separate train/validation/test seeds")
        splits = (self.train_seeds, self.validation_seeds, self.test_seeds)
        if not self.train_seeds or not self.validation_seeds:
            raise ValueError("Training and validation scenario splits must be nonempty")
        if any(len(set(seeds)) != len(seeds) for seeds in splits):
            raise ValueError("Each scenario split needs distinct seeds")
        if any(set(splits[i]) & set(splits[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Training, validation and test scenarios must be disjoint")
        if self.num_env_runners or self.num_learners or self.num_gpus_per_learner not in (0, 1):
            raise ValueError("Use one local EnvRunner and Learner, on CPU or one GPU")
        if not self.enable_rl_module_and_learner or not self.enable_env_runner_and_connector_v2:
            raise ValueError("Stackelberg requires the native RLModule/Learner API stack")
        positive = (
            "rollout_length",
            "num_epochs",
            "critic_epochs",
            "minibatch_size",
            "inner_updates",
            "response_updates",
            "uc_candidates",
            "uc_updates_per_candidate",
            "keep_checkpoints",
        )
        if any(not isinstance(getattr(self, key), int) or getattr(self, key) < 1 for key in positive):
            raise ValueError("Sampling, update, candidate and retention budgets must be positive integers")
        if self.deviation_updates < 0 or self.checkpoint_interval < 0:
            raise ValueError("Deviation and checkpoint intervals must be nonnegative")
        if (
            not math.isfinite(self.acceptance_z)
            or self.acceptance_z <= 0
            or not math.isfinite(self.min_uc_gain)
        ):
            raise ValueError("Use a positive finite confidence multiplier and a finite acceptance threshold")
        if self.stopping_patience < 0 or any(
            not math.isfinite(value) or value < 0 for value in self.stopping_tolerances.values()
        ):
            raise ValueError("Stopping patience and tolerances must be nonnegative and finite")
        for key in ("actor_lr", "critic_lr", "clip_param", "grad_clip"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("gamma", "lambda_gae", "critic_lambda"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must lie in [0, 1]")
        if self.evaluation_steps is not None and (
            not isinstance(self.evaluation_steps, int) or self.evaluation_steps < 1
        ):
            raise ValueError("evaluation_steps must be a positive integer or None")
        env = self.scenario_factory(self.train_seeds[0])
        if env.episode_mode != "finite":
            raise ValueError("Candidate selection requires finite scenarios; continuous runners support sampling and diagnostics")
        if self.evaluation_steps is not None:
            raise ValueError("Finite candidate evaluation requires complete episodes (evaluation_steps=None)")
