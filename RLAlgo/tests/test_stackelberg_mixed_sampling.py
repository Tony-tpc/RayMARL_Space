"""Segment bootstrap, fixed-policy stream mixing, and native branch continuation."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.sn_energy_market.training_audit import assert_same_state, reference_advantages
from experiments.sn_energy_market.training_trace import TrainingTrace
from raymarl.algorithms.stackelberg.algorithm import Stackelberg
from raymarl.algorithms.stackelberg.runner import (
    StackelbergEnvRunner,
    bootstrapped_returns,
    generalized_advantages,
)
from RLAlgo.tests.test_stackelberg_algorithm import configuration
from RLAlgo.tests.test_stackelberg_sampling import ToyConfig, assert_batches_equal, make_runner


@pytest.mark.parametrize("lambda_c", [0.0, 0.6, 1.0])
def test_segment_cut_bootstraps_without_cross_segment_recursion(lambda_c):
    rewards = np.array([1., 2., 500., 600.])
    values = np.array([3., 4., 5., 6.])
    next_values = np.array([4., 17., 6., 19.])
    terminals = np.array([False, False, False, True])
    ends = np.array([False, True, False, True])
    targets = bootstrapped_returns(rewards, next_values, terminals, .9, lambda_c, ends)
    expected = np.r_[bootstrapped_returns(rewards[:2], next_values[:2], terminals[:2], .9, lambda_c),
                     bootstrapped_returns(rewards[2:], next_values[2:], terminals[2:], .9, lambda_c)]
    np.testing.assert_allclose(targets, expected, rtol=0, atol=1e-12)
    advantages, _ = generalized_advantages(rewards, values, next_values, terminals, .9, .97, ends)
    np.testing.assert_allclose(advantages, reference_advantages({
        "rewards": rewards, "vf_preds": values, "next_vf_preds": next_values,
        "terminateds": terminals, "segment_end": ends,
    }, .9, .97))
    assert targets[1] == pytest.approx(2 + .9 * 17)
    assert targets[-1] == 600


def test_collection_batches_values_without_single_state_calls():
    runner = make_runner(terminal_at=3)
    weights = copy.deepcopy(runner.module.get_state())
    calls = []
    original = runner._values
    with (patch.object(runner, "_value", side_effect=AssertionError("scalar value during collection")),
          patch.object(runner, "_values", side_effect=lambda key, inputs: (calls.append((key, len(inputs))), original(key, inputs))[1])):
        batch = runner.collect(8)
    assert calls == [("uc", 9), ("consumer_0", 9)]
    assert_same_state(weights, runner.module.get_state())
    np.testing.assert_array_equal(batch["uc"]["segment_end"], [False, False, True, False, False, True, False, True])
    assert batch["uc"]["next_vf_preds"][-1] == 20


def test_mixed_collect_preserves_four_paused_streams_and_fresh_active_snapshot():
    cfg = ToyConfig(terminal_at=3)
    cfg.train_seeds = (7, 8, 9, 10)
    cfg.train_seed_groups = tuple((seed,) for seed in cfg.train_seeds)
    cfg.phase_sampling = "all_streams"
    runner = StackelbergEnvRunner(config=cfg)
    state = runner.get_state()
    original = copy.deepcopy(state)
    rng = torch.get_rng_state().clone()
    batch = runner.collect_phase(16)
    assert batch.count == 16 and runner.lane_steps == [4, 4, 4, 4]
    np.testing.assert_array_equal(batch["uc"]["stream_id"], np.repeat(np.arange(4), 4))
    assert int(batch["uc"]["segment_end"].sum()) == 8
    saved = runner.get_state()["sampling"]
    assert_same_state(saved["environment"], saved["streams"]["bank"][3]["environment"])
    assert_same_state(state, original)
    assert torch.equal(rng, torch.get_rng_state())
    runner.set_state(state)
    assert_batches_equal(batch, runner.collect_phase(16))
    for data in batch.policy_batches.values():
        np.testing.assert_allclose(data["advantages"], reference_advantages(data, cfg.gamma, cfg.lambda_gae))
    with pytest.raises(ValueError, match="divide"):
        runner.collect_phase(15)


def test_saved_value_inputs_survive_environment_buffer_reuse():
    runner = make_runner(terminal_at=None)
    state = runner.get_state()
    expected = runner.collect(4)
    runner.set_state(state)
    publish, step = runner.env.publish, runner.env.step
    buffers = []

    def announcement(*args):
        decision = publish(*args)
        buffers.extend((decision.uc_observation, decision.consumer_value_state))
        return decision

    def settlement(*args):
        result = step(*args)
        for value in buffers:
            value[:] = -12345
        return result

    with (patch.object(runner.env, "publish", side_effect=announcement),
          patch.object(runner.env, "step", side_effect=settlement)):
        observed = runner.collect(4)
    assert_batches_equal(expected, observed)


def test_mixed_multistream_deviation_contract(tmp_path):
    cfg = configuration(tmp_path).training(phase_sampling="all_streams", deviation_updates=1)
    cfg.train_seed_groups = ((2,), (3,))
    with pytest.raises(ValueError, match="deviation_updates=0"):
        cfg.validate()
    cfg.deviation_updates = 0
    cfg.validate()
    cfg.phase_sampling = "single_stream"
    cfg.deviation_updates = 1
    cfg.validate()


def test_deviation_restart_crosses_month_and_restores_original_stream():
    cfg = ToyConfig(terminal_at=3)
    cfg.train_seeds = (7, 8, 9, 10)
    cfg.train_seed_groups = ((7, 8), (9, 10))
    cfg.deviation_updates = 2
    cfg.validation_seeds = (11, 12)
    cfg.acceptance_z = 1.96
    runner = StackelbergEnvRunner(config=cfg)
    runner.begin_round(1)
    runner.collect(2)
    before = runner.get_state()
    observed = []
    fake = SimpleNamespace(config=cfg, env_runner=runner, outer_round=1, consumer_ids=("consumer_0",))
    fake._snapshot, fake._restore = runner.get_state, runner.set_state
    fake._evaluate = lambda seeds: [
        {"seed": seed, "returns": {"consumer_0": 0.0}, "score_complete": True} for seed in seeds
    ]

    def adapt(module_ids, updates, *, seed, phase):
        # Exercise the production restart and real month rollover without an optimizer.
        Stackelberg._adapt(fake, module_ids, 0, seed=seed, phase=phase)
        observed.append((runner.active_lane, runner._scenario_seed))
        batch = runner.collect_phase(cfg.rollout_length * updates)
        assert batch.count == 8
        assert runner._scenario_seed in cfg.train_seed_groups[runner.active_lane]

    fake._adapt = adapt
    try:
        result = Stackelberg.search_deviations(fake)
        assert result["searched"] and result["maximum_found_gain"] == 0.0
        assert observed == [(0, 8)]
        assert_same_state(before, runner.get_state())
        assert runner.active_lane == 1 and runner.env.t == 2
    finally:
        runner.stop()


@pytest.mark.integration
def test_native_mixed_always_branch_budget_gae_and_checkpoint(ray_runtime, tmp_path):
    torch.set_num_threads(1)
    cfg = configuration(tmp_path).training(
        phase_sampling="all_streams", deviation_updates=0, always_keep_candidate=True
    )
    cfg.train_seed_groups = ((2,), (3,))
    algo = cfg.build_algo()
    audit = TrainingTrace(algo, tmp_path, write_trajectory=False).install()
    try:
        algo.train()
        audit.assert_budget(1)
        assert len(audit.batches) == 4
        assert all(row["stream_samples"] == {"0": 2, "1": 2} for row in audit.batches)
        assert algo.env_runner.lane_steps == [4, 4]
        saved = algo.save_to_path(str(tmp_path / "checkpoint"))
        expected = algo.env_runner.collect_phase(4)
        algo.restore_from_path(saved)
        assert_batches_equal(expected, algo.env_runner.collect_phase(4))
    finally:
        audit.close()
        algo.stop()
