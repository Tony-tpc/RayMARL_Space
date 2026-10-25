"""Isolated seed-11 sampling-order / candidate-gate experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for directory in (
    ROOT,
    ROOT / "RLAlgo/src",
    ROOT / "environments/sn_energy_market/src",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
