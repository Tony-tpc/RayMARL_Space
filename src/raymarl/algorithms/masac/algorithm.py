"""MASAC native RLlib Algorithm."""

from raymarl.common.algorithm import OffPolicyAlgorithm


class MASAC(OffPolicyAlgorithm):
    @classmethod
    def get_default_config(cls):
        from .config import MASACConfig

        return MASACConfig(algo_class=cls)
