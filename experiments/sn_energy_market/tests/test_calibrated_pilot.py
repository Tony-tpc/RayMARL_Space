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


@pytest.mark.parametrize("fail", [False, True])
def test_value_auxiliary_sampling_restores_main_trajectory_even_on_error(
    tmp_path, monkeypatch, fail
):
    from types import SimpleNamespace

    from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner

    from experiments.sn_energy_market.calibrated_pilot import collect_value_data
    from RLAlgo.tests.test_stackelberg_algorithm import configuration
    from RLAlgo.tests.test_stackelberg_sampling import assert_batches_equal

    cfg = configuration(tmp_path)
    cfg.prepare_policies()
    runner = StackelbergEnvRunner(config=cfg, worker_index=0)
    runner.collect(2)
    saved = runner.get_state()
    algorithm = SimpleNamespace(config=cfg, env_runner=runner)
    audit = SimpleNamespace(phase="main")
    original = runner.collect
    if fail:

        def broken(steps):
            original(1)
            raise RuntimeError("injected auxiliary failure")

        monkeypatch.setattr(runner, "collect", broken)
        with pytest.raises(RuntimeError, match="injected"):
            collect_value_data(algorithm, (2,), audit)
        monkeypatch.setattr(runner, "collect", original)
    else:
        data, episodes = collect_value_data(algorithm, (2,), audit)
        assert episodes[0]["periods"] == 6
        assert len(data["uc"]["value_targets"]) == 6
    assert audit.phase == "main"
    assert runner.env.engine.state == saved["sampling"]["environment"].engine.state
    assert runner.num_env_steps_sampled_lifetime == 2
    actual = runner.collect(2)
    runner.set_state(saved)
    assert_batches_equal(actual, runner.collect(2))
    for seeds in ((), (11,), (21,)):
        with pytest.raises(ValueError, match="training seeds only"):
            collect_value_data(algorithm, seeds, audit)
