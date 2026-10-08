"""Ye 2021 local mixed policy and training-only attention Q. 管理网络和网络状态"""

from raymarl.common.models import ActorModule, AttentionCritic
from raymarl.common.modules import JointCriticModule
from raymarl.common.targets import soft_update


class MAACRLModule(ActorModule):
    def setup(self):
        super().setup()
        if not self.inference_only:
            self.make_target_policy()

    def update_targets(self, tau):
        soft_update(self.target_policy["encoder"], self.encoder, tau)
        soft_update(self.target_policy["head"], self.head, tau)


class MAACCriticModule(JointCriticModule):
    critic_class = AttentionCritic
