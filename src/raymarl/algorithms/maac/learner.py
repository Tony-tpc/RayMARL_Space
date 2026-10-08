"""Ye 2021 score-function update with counterfactual action marginalization."""

import torch

from raymarl.common.connectors import CRITIC_ID
from raymarl.common.learner import JointTorchLearner
from raymarl.common.targets import score_function_loss, soft_target
from raymarl.distributions import HybridDistribution


class MAACTorchLearner(JointTorchLearner):
    optimizer_type = torch.optim.RMSprop

    def compute_losses(self, *, fwd_out, batch):
        cfg = self.config
        actors, obs, actions, rewards, next_obs, terms = self.unpack_joint(batch)
        qmodule = self.module[CRITIC_ID]
        if self._phase == "critic":
            with torch.no_grad():
                draws = [a.distribution(o, target=True).draw() for a, o in zip(actors, next_obs)]
                target = soft_target(
                    rewards,
                    terms,
                    qmodule.target_critic(next_obs, [d[0] for d in draws]),
                    torch.stack([d[1] for d in draws], -1),
                    cfg.gamma,
                    cfg.temperature,
                )
            loss = (qmodule.critic(obs, actions) - target).square().mean(0).sum()
            return {CRITIC_ID: self.log_loss(CRITIC_ID, "critic_loss", loss)}
        with torch.no_grad():
            other_actions = [a.distribution(o).draw()[0] for a, o in zip(actors, obs)]
            encoded = qmodule.critic.encode_observations(obs)
        losses = {}
        for i, (mid, actor, observation) in enumerate(zip(self.actor_ids, actors, obs)):
            distribution = actor.distribution(observation)
            action, logp, _ = distribution.draw()
            with torch.no_grad():
                joint = list(other_actions)
                joint[i] = action
                qi = qmodule.critic(obs, joint, encoded=encoded)[:, i]
                baseline = torch.zeros_like(qi)
                if cfg.counterfactual:
                    k, n = cfg.baseline_samples, distribution.logits.shape[0]
                    expanded = HybridDistribution(
                        distribution.logits.unsqueeze(0).expand(k, -1, -1).reshape(k * n, -1),
                        action_space=actor.action_space,
                    )
                    features = [e.repeat(k, 1) for e in encoded]
                    joint = [a.repeat(k, 1) for a in other_actions]
                    for choice in distribution.layout.discrete_choices():
                        candidate, _, dlp = expanded.draw(discrete_choice=choice)
                        joint[i] = candidate
                        values = qmodule.critic(obs, joint, encoded=features)[:, i]
                        baseline += (dlp.exp() * values).reshape(k, n).mean(0)
            loss = score_function_loss(logp, qi, baseline, cfg.temperature)
            losses[mid] = self.log_loss(mid, "actor_loss", loss)
        return losses
