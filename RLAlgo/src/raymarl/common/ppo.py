"""Shared PPO distribution bridge; PPO optimization stays inside RLlib."""

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import ENV_CREATOR, _global_registry

from raymarl.distributions import HybridDistribution

from .models import ActorModule, LocalValue


class BasePPOModule(ActorModule, ValueFunctionAPI):
    central = False
    value_class = LocalValue

    def setup(self):
        original_space = self.observation_space
        local_space = original_space["local"] if self.central else original_space
        self.observation_space = local_space
        super().setup()
        self.observation_space = original_space
        if not self.inference_only:
            self.value = self.value_class(
                original_space["critic"] if self.central else local_space,
                self.model_config.get("hidden_dim", 64),
            )

    def get_non_inference_attributes(self):
        return ["value"]

    def _forward(self, batch, **kwargs):
        raw = batch[Columns.OBS]["local"] if self.central else batch[Columns.OBS]
        return {Columns.ACTION_DIST_INPUTS: self.head(self.encoder(self.encoder.layout.torch_batch(raw)))}

    def compute_values(self, batch, embeddings=None):
        raw = batch[Columns.OBS]["critic"] if self.central else batch[Columns.OBS]
        return self.value(raw)

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
    """Training input adapter; the policy forward pass reads only local."""

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
        if set(observations) != set(self.possible_agents):
            raise ValueError("Central value adapter requires fixed synchronous agents")
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
