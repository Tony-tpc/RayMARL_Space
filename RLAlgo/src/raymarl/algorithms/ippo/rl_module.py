"""IPPO local actor with local value."""

from raymarl.common.models import LocalValue
from raymarl.common.ppo import BasePPOModule


class IPPORLModule(BasePPOModule):
    central = False
    value_class = LocalValue
