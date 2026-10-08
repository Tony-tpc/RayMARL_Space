"""MAAC native RLlib Algorithm. 组织训练流程"""

from raymarl.common.algorithm import OffPolicyAlgorithm


class MAAC(OffPolicyAlgorithm):
    @classmethod
    def get_default_config(cls):
        from .config import MAACConfig

        return MAACConfig(algo_class=cls)
