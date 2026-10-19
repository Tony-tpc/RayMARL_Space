"""Two-stage, response-evaluated Stackelberg PPO."""

from .algorithm import Stackelberg
from .config import StackelbergConfig

__all__ = ["Stackelberg", "StackelbergConfig"]
