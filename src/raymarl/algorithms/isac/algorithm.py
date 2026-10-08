"""ISAC native RLlib Algorithm."""

from raymarl.common.algorithm import OffPolicyAlgorithm


class ISAC(OffPolicyAlgorithm):
    @classmethod
    def get_default_config(cls):
        from .config import ISACConfig

        return ISACConfig(algo_class=cls)
