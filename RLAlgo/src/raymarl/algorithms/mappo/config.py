"""MAPPO configuration and native module/learner factories."""

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from raymarl.common.config import ModelConfigMixin
from raymarl.common.ppo import CentralValueEnv

from .learner import MAPPOTorchLearner
from .rl_module import MAPPORLModule


class MAPPOConfig(ModelConfigMixin, PPOConfig):
    def __init__(self, algo_class=None):
        from .algorithm import MAPPO

        super().__init__(algo_class or MAPPO)
        self._init_outputs()
        self.normalize_actions = self.clip_actions = False

    def get_default_learner_class(self):
        return MAPPOTorchLearner

    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=MAPPORLModule)

    def environment(self, env=None, *, env_config=None, **kwargs):
        if env is None:
            return super().environment(env_config=env_config, **kwargs)
        if env is CentralValueEnv:
            return super().environment(env, env_config=env_config, **kwargs)
        return super().environment(
            CentralValueEnv, env_config={"base_env": env, "base_config": env_config or {}}, **kwargs
        )
