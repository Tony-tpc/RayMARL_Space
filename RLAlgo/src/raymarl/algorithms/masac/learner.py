"""Centralized twin-Q SAC with mixed-action policy gradients."""

import torch

from raymarl.common.connectors import CRITIC_ID
from raymarl.common.learner import JointTorchLearner
from raymarl.common.targets import soft_target


class MASACTorchLearner(JointTorchLearner):
    def compute_losses(self, *, fwd_out, batch):
        cfg = self.config
        actors, obs, actions, rewards, next_obs, terms = self.unpack_joint(batch)
        qmodule = self.module[CRITIC_ID]
        if self._phase == "critic":
            with torch.no_grad():
                draws = [a.distribution(o).draw() for a, o in zip(actors, next_obs)]
                q1, q2 = qmodule.target_critic(next_obs, [d[0] for d in draws])
                target = soft_target(
                    rewards,
                    terms,
                    torch.minimum(q1, q2),
                    torch.stack([d[1] for d in draws], -1),
                    cfg.gamma,
                    cfg.temperature,
                )
            q1, q2 = qmodule.critic(obs, actions)
            loss = ((q1 - target).square() + (q2 - target).square()).mean(0).sum()
            return {CRITIC_ID: self.log_loss(CRITIC_ID, "critic_loss", loss)}
        with torch.no_grad():
            other = [a.distribution(o).draw()[0] for a, o in zip(actors, obs)]
        losses = {}
        for i, (mid, actor, observation) in enumerate(zip(self.actor_ids, actors, obs)):
            distribution = actor.distribution(observation)
            loss = observation["current"].new_zeros(())
            for choice in distribution.layout.discrete_choices():
                action, lp, dlp = distribution.draw(reparameterize=True, discrete_choice=choice)
                joint = list(other)
                joint[i] = action
                q1, q2 = qmodule.critic(obs, joint)
                loss = loss + (dlp.exp() * (cfg.temperature * lp - torch.minimum(q1, q2)[:, i])).mean()
            losses[mid] = self.log_loss(mid, "actor_loss", loss)
        return losses
