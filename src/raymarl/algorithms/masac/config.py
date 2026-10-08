"""Explicit MASAC configuration."""

from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from raymarl.common.config import JointConfig

from .learner import MASACTorchLearner
from .rl_module import MASACCriticModule, MASACRLModule


class MASACConfig(JointConfig):
    critic_module_class = MASACCriticModule

    def __init__(self, algo_class=None):
        from .algorithm import MASAC

        super().__init__(algo_class or MASAC)

    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=MASACRLModule)

    def get_default_learner_class(self):
        return MASACTorchLearner
