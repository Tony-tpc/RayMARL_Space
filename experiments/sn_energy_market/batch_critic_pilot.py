"""Exactly one 24-batch run of separate Actor/Critic targets; test data stay reserved."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.sn_energy_market.online_critic_pilot import run_pilot

OUTPUT = ROOT / "outputs/ausgrid-batch-critic-pilot/20261020"
REFERENCE = ROOT / "outputs/ausgrid-online-critic-pilot/20261020/report.json"


if __name__ == "__main__":
    run_pilot(OUTPUT, REFERENCE, require_same_budget=True)
