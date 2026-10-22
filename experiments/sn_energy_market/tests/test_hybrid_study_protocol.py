"""Protocol checks without creating a trainer or fitting a policy."""

from types import SimpleNamespace

import pytest

from experiments.sn_energy_market.batch_critic_pilot import round_budget


def test_longer_batches_scale_sample_and_optimizer_budget():
    cfg = SimpleNamespace(
        inner_updates=2, uc_candidates=1, uc_updates_per_candidate=2,
        response_updates=2, deviation_updates=0, critic_epochs=4,
        rollout_length=64, minibatch_size=32,
    )
    old = round_budget(cfg)
    assert old == {"batches": 4, "ppo_periods": 512, "main_periods": 256, "critic_steps_per_module": 64}
    cfg.rollout_length = 192
    new = round_budget(cfg)
    assert new == {"batches": 4, "ppo_periods": 1536, "main_periods": 768, "critic_steps_per_module": 192}
    assert {key: value * 10 for key, value in new.items()} == {
        "batches": 40, "ppo_periods": 15360, "main_periods": 7680, "critic_steps_per_module": 1920,
    }
    cfg.deviation_updates = 1
    with pytest.raises(ValueError, match="deviation_updates=0"):
        round_budget(cfg)


@pytest.mark.parametrize("units", [1, 2])
def test_observer_records_phase_batch_without_changing_actions_or_rng(tmp_path, units):
    import json

    import numpy as np
    import torch
    from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner

    from experiments.sn_energy_market.hybrid_storage_study import StudyObserver
    from RLAlgo.tests.test_stackelberg_algorithm import configuration
    from RLAlgo.tests.test_stackelberg_sampling import assert_batches_equal

    cfg = configuration(tmp_path).training(rollout_length=192, inner_updates=units)
    cfg.prepare_policies()
    runner = StackelbergEnvRunner(config=cfg)
    state = runner.get_state()
    expected = runner.collect(192 * units)
    expected_rng = torch.get_rng_state().clone()
    runner.set_state(state)
    (tmp_path / "experiment.json").write_text(json.dumps({"source_sha256": {}}), encoding="utf-8")
    observer = StudyObserver(tmp_path, {})
    algorithm = SimpleNamespace(config=cfg, env_runner=runner, sampled_periods=0)
    observer.on_ready(algorithm, SimpleNamespace(phase="round_1"))
    try:
        actual = runner.collect(192 * units)
        assert_batches_equal(expected, actual)
        assert torch.equal(torch.get_rng_state(), expected_rng)
    finally:
        observer.close()
    rows = np.genfromtxt(tmp_path / "data/training-all-branches.csv", delimiter=",", names=True, encoding="utf-8-sig")
    assert len(rows) == 192 * units
    probs = np.column_stack([rows[f"prob_{m}"] for m in ("idle", "charge", "discharge")])
    np.testing.assert_allclose(probs.sum(1), 1)
    assert np.isfinite(rows["uc_value"]).all()
    np.testing.assert_allclose(rows["consumer_reward"], -rows["consumer_cost"], atol=1e-12)
