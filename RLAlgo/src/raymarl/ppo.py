"""Compatibility imports; public algorithm classes live in their own packages."""

from .algorithms.ippo import IPPOConfig
from .algorithms.mappo import MAPPOConfig
from .algorithms.mappo.rl_module import MAPPORLModule
from .common.ppo import CentralValueEnv, HybridPPOLearner

PPOModule = MAPPORLModule
__all__ = ["CentralValueEnv", "HybridPPOLearner", "IPPOConfig", "MAPPOConfig", "PPOModule"]
