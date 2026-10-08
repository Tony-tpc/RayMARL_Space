"""Training-only module ownership for joint critics."""

import copy

from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

from .targets import soft_update


class JointCriticModule(TorchRLModule):
    critic_class = None

    def setup(self):
        self.actor_ids = tuple(self.model_config["actor_ids"])
        self.critic = self.critic_class(
            [self.observation_space[mid] for mid in self.actor_ids],
            [self.action_space[mid] for mid in self.actor_ids],
            self.model_config.get("hidden_dim", 64),
        )
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)

    def _forward(self, batch, **kwargs):
        return {}

    def update_targets(self, tau):
        soft_update(self.target_critic, self.critic, tau)
