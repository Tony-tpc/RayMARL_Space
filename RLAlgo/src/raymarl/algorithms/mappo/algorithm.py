"""MAPPO uses RLlib's native PPO runtime."""

from ray.rllib.algorithms.ppo import PPO

from raymarl.common.algorithm import NativeAlgorithmMixin


class MAPPO(NativeAlgorithmMixin, PPO):
    @classmethod
    def get_default_config(cls):
        from .config import MAPPOConfig

        return MAPPOConfig(algo_class=cls)
