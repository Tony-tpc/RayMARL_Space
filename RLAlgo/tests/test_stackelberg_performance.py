"""Execution optimizations must preserve trajectories, RNG and snapshot isolation."""
import copy

import numpy as np
import pytest
import torch
from experiments.sn_energy_market.training_audit import assert_same_state
from RLAlgo.tests.test_stackelberg_algorithm import configuration
from RLAlgo.tests.test_stackelberg_policy import bounds_tensor, legal_bounds, module_spec

from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner, random_state, restore_random_state


@pytest.mark.parametrize("role", ["uc", "consumer"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_action_only_preserves_draw_and_rng(role, deterministic):
    module = module_spec(role).build()
    obs = torch.randn(32, 7)
    bounds = (torch.tensor([[10., 8., 5., 5., 4., 3.]] * 32, dtype=torch.float64)
              if role == "uc" else bounds_tensor(legal_bounds(), 32))
    # Exercise collapsed coordinates and unavailable storage modes too.
    bounds[:8, 0] = 0
    if role == "uc":
        bounds[:8, 2:5] = 0
    before = random_state()
    with torch.no_grad():
        full = module.draw(obs, bounds, deterministic)
        expected_rng = random_state()
        restore_random_state(before)
        action = module.draw_action(obs, bounds, deterministic)
    expected = (torch.cat((full["storage_mode"][:, None], full["raw_latents"]), -1)
                if role == "uc" else full["actions"])
    torch.testing.assert_close(action, expected, atol=0, rtol=0)
    assert_same_state(expected_rng, random_state())


@pytest.mark.parametrize("deterministic", [False, True])
def test_complete_evaluation_and_snapshot_isolation(tmp_path, deterministic):
    config = configuration(tmp_path)
    config.prepare_policies()
    runner = StackelbergEnvRunner(config=config)
    runner.collect(3)
    snapshot = runner.get_state()
    untouched = copy.deepcopy(snapshot)
    expected = runner.evaluate((11, 12), deterministic=deterministic)
    actual = runner.evaluate((11, 12), deterministic=deterministic, actions_only=True)
    assert_same_state(expected, actual)
    assert_same_state(snapshot, runner.get_state())
    runner.set_state(snapshot)
    runner.collect(8)
    assert_same_state(untouched, snapshot)
    runner.set_state(snapshot)
    first = runner.collect(4)
    runner.set_state(snapshot)
    second = runner.collect(4)
    for key, rows in first.policy_batches.items():
        for name, values in rows.items():
            np.testing.assert_array_equal(values, second[key][name])
