"""Deterministic continuous actors and joint action-value critic."""

import torch
from ray.rllib.core.columns import Columns
from torch import nn

from raymarl.common.models import ActorModule, JointQCritic
from raymarl.common.modules import JointCriticModule
from raymarl.common.targets import soft_update


class MADDPGRLModule(ActorModule):
    def setup(self):
        super().setup()
        if not self.layout.is_continuous:
            raise ValueError("Standard MADDPG requires continuous Box actions")
        self.head = nn.Linear(self.encoder.feature_dim, self.layout.feature_dim)
        if not self.inference_only:
            self.make_target_policy()

    def policy_action(self, obs, *, target=False):
        policy = self.target_policy if target else {"encoder": self.encoder, "head": self.head}
        return torch.tanh(policy["head"](policy["encoder"](obs)))

    def _forward(self, batch, **kwargs):
        obs = self.encoder.layout.torch_batch(batch[Columns.OBS])
        return {Columns.ACTIONS: self.layout.decode_features(self.policy_action(obs))}

    def _forward_exploration(self, batch, **kwargs):
        with torch.no_grad():
            obs = self.encoder.layout.torch_batch(batch[Columns.OBS])
            action = self.policy_action(obs)
            action = (
                action + torch.randn_like(action) * self.model_config.get("exploration_std", 0.1)
            ).clamp(-1, 1)
            return {Columns.ACTIONS: self.layout.decode_features(action)}

    def update_targets(self, tau):
        soft_update(self.target_policy["encoder"], self.encoder, tau)
        soft_update(self.target_policy["head"], self.head, tau)


class MADDPGCriticModule(JointCriticModule):
    critic_class = JointQCritic
