"""Month-interior starts preserve the actual billing/state history and terminal."""

import copy
import pickle

import numpy as np
import pytest
from snmarket.contracts import ConsumerAction, Phase

from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)
from experiments.sn_energy_market.tests.test_profile_dataset import make_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = make_dataset(tmp_path_factory.mktemp("training_dates"))
    (path.parent / "test.npz").unlink()
    return ProfileDataset(path, splits=("train", "validation"))


def passive_step(env):
    env.publish([1, 0, 0, 0, 0, 0])
    action = ConsumerAction(
        curtail_kw=0,
        signed_shift_kw=0,
        der_preference_fraction=1,
        der_bid_per_kwh=env.config.market.der_reference_rate_per_kwh,
    )
    return env.step(dict.fromkeys(env.consumer_ids, action))


def test_training_seed_schedule_covers_months_and_interior_dates(dataset):
    factory = DatasetEnvFactory(dataset, training_start_days=(1, 11, 21, -1))
    contract = factory.get_contract()
    assert len(factory.seed_splits["train"]) == 48
    scheduled = [contract["seed_windows"][str(seed)] for seed in factory.seed_splits["train"]]
    assert len({row["name"] for row in scheduled[:12]}) == 12
    assert {row["start_day"] for row in scheduled[:12]} >= {1, 11, 21, 30, 31}
    assert len(set(factory.seed_splits["train"])) == 48
    # Three equally sized update budgets can now sample three seasons/date strata.
    expected = {43: "2010-07-11", 56: "2010-10-21", 69: "2011-01-31"}
    for seed, date in expected.items():
        assert contract["seed_windows"][str(seed)]["sampling_start_datetime"].startswith(date)
    assert len({row["sampling_start_datetime"] for key, row in contract["seed_windows"].items()
                if int(key) in factory.seed_splits["train"]}) == 48
    assert not factory._initial_snapshots


def test_default_training_starts_at_month_beginning_without_rule_history(dataset):
    from experiments.sn_energy_market.train import TRAINING_START_DAYS

    assert TRAINING_START_DAYS == (1,)
    factory = DatasetEnvFactory(dataset, training_start_days=TRAINING_START_DAYS)
    assert len(factory.seed_splits["train"]) == 12
    for seed in factory.seed_splits["train"]:
        env = factory(seed)
        env.reset(seed)
        assert env.engine.state.period_id == 0
        assert env.engine.state.timestamp.day == 1
        assert not env.engine.records
    assert not factory._initial_snapshots
    assert "test" not in factory.seed_splits
    for seed in factory.seed_splits["train"]:
        assert factory.profiles_for_seed(seed).metadata["split"] == "train"
    assert not factory._initial_snapshots


def test_validation_seeds_and_initial_states_are_unchanged(dataset):
    ordinary = DatasetEnvFactory(dataset)
    diverse = DatasetEnvFactory(dataset, training_start_days=(1, 11, 21, -1))
    assert diverse.seed_splits["validation"] == ordinary.seed_splits["validation"]
    for seed in diverse.seed_splits["validation"]:
        expected, actual = ordinary(seed), diverse(seed)
        np.testing.assert_array_equal(expected.reset(seed), actual.reset(seed))
        assert actual.engine.state.timestamp.day == 1
        assert actual.engine.state.period_id == 0
        assert actual.config == expected.config
    assert not diverse._initial_snapshots


def test_warm_start_contains_actual_monthly_energy_history_and_reproducible_state(dataset):
    factory = DatasetEnvFactory(dataset, training_start_days=(11,))
    before_contract = factory.get_contract()
    env = factory(42)
    observation = env.reset(42)
    state = env.engine.state
    assert state.period_id == 10 * 48
    assert state.timestamp.day == 11
    assert state.ess_energy_kwh == env.config.system.ess_initial_energy_kwh
    records = env.engine.records
    assert len(records) == state.period_id
    for consumer in state.consumers:
        actual_uc = sum(record.market.allocations[consumer.consumer_id].uc_kw for record in records) * 0.5
        assert consumer.monthly_uc_energy_kwh == pytest.approx(actual_uc)
        assert consumer.monthly_uc_energy_kwh > 100
        assert consumer.tasks == ()
    passive_step(env)
    np.testing.assert_array_equal(env.reset(42), observation)
    assert env.engine.state == state
    assert factory.get_contract() == before_contract
    another = factory(42)
    np.testing.assert_array_equal(another.reset(42), observation)
    passive_step(another)
    assert env.engine.state == state
    with pytest.raises(ValueError, match="reset overrides"):
        env.reset(42, options={"ess_energy_kwh": 0})


def test_final_day_start_keeps_original_month_end_terminal(dataset):
    factory = DatasetEnvFactory(dataset, training_start_days=(-1,))
    env = factory(42)
    env.reset(42)
    assert env.engine.state.period_id == 30 * 48
    assert env.config.system.horizon_steps == 31 * 48
    assert env.engine.state.timestamp.day == 31
    for step in range(48):
        result = passive_step(env)
        assert result.terminated == (step == 47)
    assert env.phase == Phase.TERMINATED
    assert env.engine.state.timestamp.isoformat() == "2010-08-01T00:00:00+10:00"
    assert len(env.engine.records) == env.config.system.horizon_steps
    assert all(c.monthly_uc_energy_kwh == 0 for c in env.engine.state.consumers)
    env.reset(42)
    assert env.phase == Phase.UC_TURN
    assert env.engine.state.timestamp.day == 31


def test_initial_snapshot_cache_is_not_copied_or_serialized(dataset):
    factory = DatasetEnvFactory(dataset, training_start_days=(2,))
    env = factory(42)
    observation = env.reset(42)
    assert set(factory._initial_snapshots) == {42}
    for restored in (copy.deepcopy(factory), pickle.loads(pickle.dumps(factory))):
        assert not restored._initial_snapshots
        assert restored.get_contract() == factory.get_contract()
        np.testing.assert_array_equal(restored(42).reset(42), observation)


@pytest.mark.parametrize("days", [(), (0,), (1, 1), (32,), (-32,)])
def test_invalid_training_start_dates_are_rejected(dataset, days):
    with pytest.raises(ValueError, match="start day"):
        DatasetEnvFactory(dataset, training_start_days=days)
