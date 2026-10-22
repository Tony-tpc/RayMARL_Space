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
    assert old == {"batches": 8, "ppo_periods": 512, "main_periods": 256, "critic_steps_per_module": 64}
    cfg.rollout_length = 192
    new = round_budget(cfg)
    assert new == {"batches": 8, "ppo_periods": 1536, "main_periods": 768, "critic_steps_per_module": 192}
    cfg.deviation_updates = 1
    with pytest.raises(ValueError, match="deviation_updates=0"):
        round_budget(cfg)
