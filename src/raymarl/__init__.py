"""Importing raymarl never starts Ray or loads environment data."""

from .algorithms import ISACConfig, JointAlgorithm, MAACConfig, MADDPGConfig, MASACConfig
from .ppo import IPPOConfig, MAPPOConfig

__version__ = "0.1.0"


def register_algorithms():
    """Explicit opt-in registration with Ray Tune."""
    from ray.tune.registry import register_trainable

    for name, config in (
        ("MAAC", MAACConfig),
        ("MASAC", MASACConfig),
        ("MADDPG", MADDPGConfig),
        ("ISAC", ISACConfig),
    ):
        cls = type(name, (JointAlgorithm,), {"get_default_config": classmethod(lambda c, f=config: f(c))})
        register_trainable(name, cls)
    from ray.rllib.algorithms.ppo import PPO

    register_trainable(
        "MAPPO",
        type("MAPPO", (PPO,), {"get_default_config": classmethod(lambda cls: MAPPOConfig(algo_class=cls))}),
    )
    register_trainable(
        "IPPO",
        type("IPPO", (PPO,), {"get_default_config": classmethod(lambda cls: IPPOConfig(algo_class=cls))}),
    )


__all__ = [
    "IPPOConfig",
    "ISACConfig",
    "MAACConfig",
    "MADDPGConfig",
    "MAPPOConfig",
    "MASACConfig",
    "register_algorithms",
]
