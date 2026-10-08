"""MADDPG native RLlib Algorithm."""

from raymarl.common.algorithm import OffPolicyAlgorithm


class MADDPG(OffPolicyAlgorithm):
    @classmethod
    def get_default_config(cls):
        from .config import MADDPGConfig

        return MADDPGConfig(algo_class=cls)
