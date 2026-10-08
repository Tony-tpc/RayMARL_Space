"""IPPO configuration and native module/learner factories."""

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from raymarl.common.config import ModelConfigMixin

from .learner import IPPOTorchLearner
from .rl_module import IPPORLModule


class IPPOConfig(ModelConfigMixin, PPOConfig):
    def __init__(self, algo_class=None):
        from .algorithm import IPPO

        super().__init__(algo_class or IPPO)
        self._init_outputs()
        self.normalize_actions = self.clip_actions = False

    def get_default_learner_class(self):
        return IPPOTorchLearner

    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=IPPORLModule)
