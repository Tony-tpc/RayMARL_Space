"""IPPO uses RLlib's native PPO runtime."""

from ray.rllib.algorithms.ppo import PPO

from raymarl.common.algorithm import NativeAlgorithmMixin


class IPPO(NativeAlgorithmMixin, PPO):
    @classmethod
    def get_default_config(cls):
        from .config import IPPOConfig

        return IPPOConfig(algo_class=cls)
