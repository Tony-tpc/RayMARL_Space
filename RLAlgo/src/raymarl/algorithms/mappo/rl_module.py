"""MAPPO local actor with centralized value."""

from raymarl.common.models import CentralValue
from raymarl.common.ppo import BasePPOModule


class MAPPORLModule(BasePPOModule):
    central = True
    value_class = CentralValue
