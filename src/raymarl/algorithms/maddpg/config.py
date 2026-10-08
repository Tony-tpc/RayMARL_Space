"""Explicit MADDPG configuration."""

from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from raymarl.common.config import JointConfig

from .learner import MADDPGTorchLearner
from .rl_module import MADDPGCriticModule, MADDPGRLModule


class MADDPGConfig(JointConfig):
    critic_module_class = MADDPGCriticModule

    def __init__(self, algo_class=None):
        from .algorithm import MADDPG

        super().__init__(algo_class or MADDPG)
        self.temperature = 0.0

    def get_default_rl_module_spec(self):
        return RLModuleSpec(
            module_class=MADDPGRLModule, model_config={"exploration_std": self.exploration_std}
        )

    def get_default_learner_class(self):
        return MADDPGTorchLearner
