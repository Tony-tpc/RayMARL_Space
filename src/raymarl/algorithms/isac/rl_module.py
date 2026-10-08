"""Independent SAC policies with private local twin Q networks."""

import copy

from raymarl.common.models import ActorModule, TwinLocalQCritic
from raymarl.common.targets import soft_update


class ISACRLModule(ActorModule):
    def setup(self):
        super().setup()
        if not self.inference_only:
            self.critic = TwinLocalQCritic(
                self.observation_space, self.action_space, self.model_config.get("hidden_dim", 64)
            )
            self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)

    def get_non_inference_attributes(self):
        return ["critic", "target_critic"]

    def update_targets(self, tau):
        soft_update(self.target_critic, self.critic, tau)
