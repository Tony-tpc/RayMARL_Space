import numpy as np
import pytest

from experiments.sn_energy_market.calibrated_pilot import terminal_returns
from experiments.sn_energy_market.train import model_settings


def test_mc_targets_use_real_terminal_and_original_reward_units():
    rewards = np.array([1.0, 2.0, 3.0, 4.0])
    targets = terminal_returns(rewards, np.array([False, True, False, True]), 0.5)
    np.testing.assert_array_equal(targets, [2.0, 2.0, 5.0, 4.0])
    with pytest.raises(AssertionError):
        terminal_returns(rewards, np.zeros(4, dtype=bool), 0.5)


def test_experiment_priors_keep_six_stochastic_controls():
    config = model_settings(32)
    assert len(config["uc_initial_mean"]) == len(config["uc_initial_std"]) == 6
    assert all(value > 0 for value in config["uc_initial_std"])
    assert config["uc_initial_mean"][0] > 0.9
    assert config["uc_initial_mean"][1] < 0.05
    assert set(config["value_scale"]) == {"uc", "consumer"}
