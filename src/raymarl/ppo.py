"""PPO configuration reuse with a training-only centralized value input."""

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import ENV_CREATOR, _global_registry
from torch import nn

from .distributions import HybridDistribution
from .networks import ActorModule, Encoder
from .spaces import ObservationLayout


class PPOModule(ActorModule, ValueFunctionAPI):
    def setup(self):
        self.central = isinstance(self.observation_space, gym.spaces.Dict) and "critic" in (
            self.observation_space.spaces
        )
        local_space = self.observation_space["local"] if self.central else self.observation_space
        hidden = self.model_config.get("hidden_dim", 64)
        self.encoder = Encoder(local_space, hidden)
        from .spaces import ActionLayout

        self.layout = ActionLayout(self.action_space)
        self.head = nn.Linear(hidden, self.layout.param_dim)
        self.value_layout = ObservationLayout(
            self.observation_space["critic"] if self.central else local_space
        )
        self.value_encoder = Encoder(
            self.observation_space["critic"] if self.central else local_space, hidden
        )
        self.value_head = nn.Linear(hidden, 1)

    def _forward(self, batch, **kwargs):
        raw = batch[Columns.OBS]["local"] if self.central else batch[Columns.OBS]
        obs = self.encoder.layout.torch_batch(raw)
        return {Columns.ACTION_DIST_INPUTS: self.head(self.encoder(obs))}

    def compute_values(self, batch, embeddings=None):
        raw = batch[Columns.OBS]["critic"] if self.central else batch[Columns.OBS]
        return self.value_head(self.value_encoder(self.value_layout.torch_batch(raw))).squeeze(-1)

    def _forward_exploration(self, batch, **kwargs):
        output = self._forward(batch)
        distribution = HybridDistribution(output[Columns.ACTION_DIST_INPUTS], action_space=self.action_space)
        action, lp, _, latent = distribution.draw(return_latents=True)
        output.update(
            {
                Columns.ACTIONS: distribution.layout.decode_features(action),
                Columns.ACTION_LOGP: lp,
                "action_latents": latent,
            }
        )
        return output

    def get_train_action_dist_cls(self):
        return HybridDistribution.get_partial_dist_cls(
            action_space=self.action_space, action_latents=getattr(self, "_loss_latents", None)
        )


class HybridPPOLearner(PPOTorchLearner):
    """Reuse RLlib PPO loss while retaining pre-Tanh samples in its log density."""

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id].unwrapped()
        module._loss_latents = batch.get("action_latents")
        try:
            return super().compute_loss_for_module(
                module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
            )
        finally:
            module._loss_latents = None


class CentralValueEnv(MultiAgentEnv):
    """Generic I/O adapter. Actor module reads only the local field."""

    def __init__(self, config):
        base = config["base_env"]
        creator = _global_registry.get(ENV_CREATOR, base) if isinstance(base, str) else base
        self.base = creator(config.get("base_config", {}))
        super().__init__()
        self.possible_agents = list(self.base.possible_agents)
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.possible_agents)
        self.action_spaces = {a: self.base.get_action_space(a) for a in self.possible_agents}
        self.original = {a: self.base.get_observation_space(a) for a in self.possible_agents}
        total = sum(gym.spaces.utils.flatdim(s) for s in self.original.values())
        critic = gym.spaces.Box(-np.inf, np.inf, (total,), np.float32)
        self.observation_spaces = {
            a: gym.spaces.Dict({"local": s, "critic": critic}) for a, s in self.original.items()
        }

    def _wrap(self, observations):
        state = np.concatenate(
            [gym.spaces.utils.flatten(self.original[a], observations[a]) for a in self.possible_agents]
        ).astype(np.float32)
        return {a: {"local": value, "critic": state.copy()} for a, value in observations.items()}

    def reset(self, *, seed=None, options=None):
        obs, infos = self.base.reset(seed=seed, options=options)
        return self._wrap(obs), infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.base.step(actions)
        return self._wrap(obs), rewards, terms, truncs, infos

    def close(self):
        self.base.close()


class IPPOConfig(PPOConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Policies already return the environment's physical Box bounds.
        self.normalize_actions = False
        self.clip_actions = False

    def get_default_learner_class(self):
        return HybridPPOLearner

    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=PPOModule)

    def build_algo(self, env=None, use_copy=True):
        from pathlib import Path

        from ray.rllib.algorithms.ppo import PPO
        from ray.train._internal.storage import StorageContext

        config = self.copy(copy_frozen=False) if use_copy else self
        if env is not None:
            config.environment(env)
        storage = StorageContext(
            storage_path=(Path.cwd() / "ray_results").as_posix(),
            experiment_dir_name="raymarl",
            trial_dir_name=type(config).__name__,
        )
        return PPO(config=config, storage=storage)


class MAPPOConfig(IPPOConfig):
    def environment(self, env=None, *, env_config=None, **kwargs):
        if env is None:
            return super().environment(env_config=env_config, **kwargs)
        return super().environment(
            env=CentralValueEnv, env_config={"base_env": env, "base_config": env_config or {}}, **kwargs
        )
