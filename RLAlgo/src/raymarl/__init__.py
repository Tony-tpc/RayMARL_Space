"""Importing raymarl never starts Ray or loads environment data."""

from .algorithms import (
    IPPO,
    ISAC,
    MAAC,
    MADDPG,
    MAPPO,
    MASAC,
    IPPOConfig,
    ISACConfig,
    MAACConfig,
    MADDPGConfig,
    MAPPOConfig,
    MASACConfig,
)

__version__ = "0.2.0"


def register_algorithms():
    """Explicit opt-in registration of importable native Algorithm classes."""
    from ray.tune.registry import register_trainable

    for cls in (MAAC, MASAC, MADDPG, ISAC, MAPPO, IPPO):
        register_trainable(cls.__name__, cls)


__all__ = [
    "IPPO",
    "ISAC",
    "MAAC",
    "MADDPG",
    "MAPPO",
    "MASAC",
    "IPPOConfig",
    "ISACConfig",
    "MAACConfig",
    "MADDPGConfig",
    "MAPPOConfig",
    "MASACConfig",
    "register_algorithms",
]
