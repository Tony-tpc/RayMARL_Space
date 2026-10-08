"""Deterministic policy gradient with other replay actions held fixed."""

import torch

from raymarl.common.connectors import CRITIC_ID
from raymarl.common.learner import JointTorchLearner


class MADDPGTorchLearner(JointTorchLearner):
    def compute_losses(self, *, fwd_out, batch):
        actors, obs, actions, rewards, next_obs, terms = self.unpack_joint(batch)
        qmodule = self.module[CRITIC_ID]
        if self._phase == "critic":
            with torch.no_grad():
                target_actions = [a.policy_action(o, target=True) for a, o in zip(actors, next_obs)]
                target = rewards + self.config.gamma * (1 - terms) * qmodule.target_critic(
                    next_obs, target_actions
                )
            loss = (qmodule.critic(obs, actions) - target).square().mean(0).sum()
            return {CRITIC_ID: self.log_loss(CRITIC_ID, "critic_loss", loss)}
        losses = {}
        for i, (mid, actor, observation) in enumerate(zip(self.actor_ids, actors, obs)):
            joint = [a.detach() for a in actions]
            joint[i] = actor.policy_action(observation)
            loss = -qmodule.critic(obs, joint)[:, i].mean()
            losses[mid] = self.log_loss(mid, "actor_loss", loss)
        return losses
