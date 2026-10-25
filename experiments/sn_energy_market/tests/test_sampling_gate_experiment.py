"""Factor isolation, private RNG and paused physical trajectories on real months."""

from types import SimpleNamespace

from raymarl.algorithms.stackelberg.runner import random_state

from experiments.sn_energy_market.sampling_gate.audit import ExperimentAudit
from experiments.sn_energy_market.sampling_gate.runtime import (
    ExperimentRunner,
    configuration,
    digest_tree,
)


def test_only_order_and_gate_change_between_four_groups(tmp_path):
    contracts = {g: configuration(tmp_path / g, g).learning_contract() for g in "ABCD"}
    base = contracts["A"]
    assert configuration(tmp_path / "gpu", "A").num_gpus_per_learner == 1
    for contract in contracts.values():
        assert {k for k in base if base[k] != contract[k]} <= {
            "acceptance_z",
            "experiment",
            "train_seed_groups",
        }
        assert contract["gamma"] == 0.99
        assert contract["rollout_length"] * contract["inner_updates"] == 384
        assert contract["seed"] == 11
        assert contract["validation_seeds"] == [10042, 10048]
        assert contract["test_seeds"] == []


def test_paused_stream_preserves_state_and_does_not_consume_learner_rng(tmp_path):
    cfg = configuration(tmp_path, "C")
    cfg.prepare_policies()
    runner = ExperimentRunner(config=cfg)
    learner_rng = digest_tree(random_state())
    runner.collect(8)
    saved = runner.get_state()
    runner.begin_round(1)
    runner.collect(7)
    assert runner.lane_steps == [8, 7, 0, 0]
    runner.begin_round(4)
    assert runner.env.engine.state.period_id == 8
    expected = digest_tree(runner.env.get_state())
    runner.set_state(saved)
    assert digest_tree(runner.env.get_state()) == expected
    assert digest_tree(random_state()) == learner_rng
    runner.stop()


def test_action_only_gate_evaluation_is_not_a_learning_trajectory(tmp_path):
    cfg = configuration(tmp_path, "A")
    cfg.prepare_policies()
    runner = ExperimentRunner(config=cfg)
    algorithm = SimpleNamespace(
        env_runner=runner,
        _adapt=lambda *args, **kwargs: None,
        learner_group=SimpleNamespace(_learner=SimpleNamespace(_named_optimizers={})),
    )
    audit = ExperimentAudit(algorithm, tmp_path).install()
    try:
        # The previous learning phase remains response when the real gate runs.
        audit.phase = "response"
        before = digest_tree(runner.get_state())
        rows = runner.evaluate((10042,), steps=2, actions_only=True)
        assert rows[0]["physical_periods"] == 2
        assert audit.physical == []
        assert audit.state() == {"steps": {}, "batches": []}
        assert not audit.collecting
        assert digest_tree(runner.get_state()) == before
    finally:
        audit.close()
        runner.stop()
