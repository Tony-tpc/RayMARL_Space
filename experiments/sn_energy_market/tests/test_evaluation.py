import numpy as np
import pytest

from experiments.sn_energy_market.evaluation import terminal_returns
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


@pytest.mark.parametrize("fail", [False, True])
def test_frozen_month_validation_restores_sampling_and_rng(tmp_path, monkeypatch, fail):
    from dataclasses import replace
    from types import SimpleNamespace

    from experiments.sn_energy_market import evaluation
    from experiments.sn_energy_market.training_env import MarketEnvFactory
    from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner
    from RLAlgo.tests.test_stackelberg_algorithm import configuration
    from RLAlgo.tests.test_stackelberg_sampling import assert_batches_equal

    cfg = configuration(tmp_path)
    physical = cfg.scenario_factory.config
    cfg.scenario_factory = MarketEnvFactory(
        replace(physical, system=replace(physical.system, horizon_steps=100))
    )
    cfg.prepare_policies()
    runner = StackelbergEnvRunner(config=cfg, worker_index=0)
    runner.collect(2)
    saved = runner.get_state()
    monkeypatch.setattr(evaluation, "VALIDATION_SEEDS", (11,))
    original = runner._period
    if fail:
        def broken():
            original()
            raise RuntimeError("injected validation failure")

        monkeypatch.setattr(runner, "_period", broken)
        with pytest.raises(RuntimeError, match="injected"):
            evaluation.full_value_validation(SimpleNamespace(env_runner=runner))
        monkeypatch.setattr(runner, "_period", original)
    else:
        report = evaluation.full_value_validation(SimpleNamespace(env_runner=runner))
        assert report["periods"] == 100
        assert report["episodes"][0]["true_terminal"]
    assert runner.env.engine.state == saved["sampling"]["environment"].engine.state
    assert runner.num_env_steps_sampled_lifetime == 2
    actual = runner.collect(2)
    runner.set_state(saved)
    assert_batches_equal(actual, runner.collect(2))
