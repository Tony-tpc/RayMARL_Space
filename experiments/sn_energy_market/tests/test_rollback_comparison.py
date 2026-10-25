"""The ablation changes candidate retention, while preserving paired-gain diagnostics."""

from types import SimpleNamespace

import pytest

from experiments.sn_energy_market.rollback_comparison import RollbackComparison


@pytest.mark.parametrize("always,expected", [(False, 0), (True, 1)])
def test_negative_candidate_is_only_retained_in_the_always_arm(always, expected):
    algorithm = object.__new__(RollbackComparison)
    algorithm.config = SimpleNamespace(acceptance_z=1.0, min_uc_gain=0.0, always_keep_candidate=always)
    rows = [[{"seed": seed, "score_complete": True, "returns": {"uc": score}}
             for seed in (11, 12)] for score in (2.0, 1.0)]
    selected, gains = algorithm._select_candidate(rows)
    assert selected == expected
    assert algorithm.gate_selected == 0
    assert gains[1]["mean"] == gains[1]["lower"] == -1.0
