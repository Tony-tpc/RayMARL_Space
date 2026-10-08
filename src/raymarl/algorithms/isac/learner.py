"""Standard independent SAC: private twin Q, no counterfactual baseline."""

import torch
from ray.rllib.core.columns import Columns

from raymarl.common.learner import PhasedTorchLearner
from raymarl.common.targets import soft_target


class ISACTorchLearner(PhasedTorchLearner):
    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id]
        obs = module.encoder.layout.torch_batch(batch[Columns.OBS])
        next_obs = module.encoder.layout.torch_batch(batch[Columns.NEXT_OBS])
        if self._phase == "critic":
            with torch.no_grad():
                distribution = module.distribution(next_obs)
                value = batch[Columns.REWARDS].new_zeros(batch[Columns.REWARDS].shape)
                for choice in distribution.layout.discrete_choices():
                    action, lp, dlp = distribution.draw(discrete_choice=choice)
                    q1, q2 = module.target_critic(next_obs, action)
                    value += dlp.exp() * (torch.minimum(q1, q2) - config.temperature * lp)
                target = soft_target(
                    batch[Columns.REWARDS],
                    batch[Columns.TERMINATEDS].float(),
                    value,
                    torch.zeros_like(value),
                    config.gamma,
                    0,
                )
            action = module.layout.encode(batch[Columns.ACTIONS], self._device)
            q1, q2 = module.critic(obs, action)
            loss = ((q1 - target).square() + (q2 - target).square()).mean()
            self.metrics.log_value((module_id, "critic_loss"), loss.detach(), window=1)
            return loss
        distribution = module.distribution(obs)
        loss = batch[Columns.REWARDS].new_zeros(())
        for choice in distribution.layout.discrete_choices():
            action, lp, dlp = distribution.draw(reparameterize=True, discrete_choice=choice)
            q1, q2 = module.critic(obs, action)
            loss += (dlp.exp() * (config.temperature * lp - torch.minimum(q1, q2))).mean()
        self.metrics.log_value((module_id, "actor_loss"), loss.detach(), window=1)
        return loss
