"""Ye 2021 configuration; unspecified paper parameters remain explicit defaults. 配置参数、组装组件 """

from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from raymarl.common.config import JointConfig

from .learner import MAACTorchLearner
from .rl_module import MAACCriticModule, MAACRLModule


class MAACConfig(JointConfig):
    critic_module_class = MAACCriticModule

    def __init__(self, algo_class=None):
        from .algorithm import MAAC

        super().__init__(algo_class or MAAC)
        self.baseline_samples = 8
        self.counterfactual = True

    def training(self, **kwargs):
        for key in ("baseline_samples", "counterfactual"):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        return super().training(**kwargs)

    def validate(self):
        super().validate()
        if not isinstance(self.baseline_samples, int) or self.baseline_samples < 1:
            raise ValueError("baseline_samples must be positive")

    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=MAACRLModule)

    def get_default_learner_class(self):
        return MAACTorchLearner
