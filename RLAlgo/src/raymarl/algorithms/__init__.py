"""Each algorithm owns its config, module and learner."""

from .ippo import IPPO, IPPOConfig
from .isac import ISAC, ISACConfig
from .maac import MAAC, MAACConfig
from .maddpg import MADDPG, MADDPGConfig
from .mappo import MAPPO, MAPPOConfig
from .masac import MASAC, MASACConfig

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
]
