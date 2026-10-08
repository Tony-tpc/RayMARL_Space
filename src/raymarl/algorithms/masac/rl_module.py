"""Stochastic local policy with a training-only joint twin Q."""

from raymarl.common.models import ActorModule, TwinJointQCritic
from raymarl.common.modules import JointCriticModule


class MASACRLModule(ActorModule):
    pass


class MASACCriticModule(JointCriticModule):
    critic_class = TwinJointQCritic
