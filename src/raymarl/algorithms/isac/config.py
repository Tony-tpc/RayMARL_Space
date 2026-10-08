"""Explicit ISAC configuration."""

from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from raymarl.common.config import OffPolicyConfig

from .learner import ISACTorchLearner
from .rl_module import ISACRLModule


class ISACConfig(OffPolicyConfig):
    def __init__(self, algo_class=None):
        from .algorithm import ISAC

        super().__init__(algo_class or ISAC)

    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=ISACRLModule)

    def get_default_learner_class(self):
        return ISACTorchLearner
