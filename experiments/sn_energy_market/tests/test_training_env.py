import json
import pickle
from dataclasses import replace

import numpy as np
import pytest
from snmarket import (
    ConsumerState,
    EnvironmentConfig,
    Phase,
    ProfileExhaustedError,
    ShiftTask,
)
from snmarket.action import map_consumer_controls

from experiments.sn_energy_market.training_env import (
    MarketEnvFactory,
    MarketTrainingEnv,
    scenario_profiles,
)


def config(*, mode="finite", steps=8):
    base = EnvironmentConfig()
    return replace(
        base,
        system=replace(
            base.system,
            num_consumers=2,
            horizon_steps=steps,
            episode_mode=mode,
            start_datetime="2026-01-01T17:00:00+08:00",
        ),
        market=replace(base.market, shift_deadline_steps=2),
    ).validate()


def consumer_actions(env, values=(0.2, 0.4, 0.5, 0.6)):
    return {
        key: map_consumer_controls(values, bounds)
        for key, bounds in env.engine.local_bounds().items()
    }


def test_full_episode_inventory_value_removes_free_initial_energy_gain():
    from snmarket import Profiles

    from experiments.sn_energy_market.train import inventory_valued_config

    base = EnvironmentConfig()
    cfg = inventory_valued_config(replace(
        base,
        system=replace(base.system, num_consumers=1, horizon_steps=2,
                       grid_import_max_kw=4, ess_energy_capacity_kwh=10,
                       ess_initial_energy_kwh=5, ess_discharge_max_kw=5),
        market=replace(base.market, dr_event_intervals=()),
    ))
    assert cfg.rewards.terminal_energy_value_per_kwh == pytest.approx(0.3135)
    explicit = replace(cfg, rewards=replace(cfg.rewards, terminal_energy_value_per_kwh=0.7))
    assert inventory_valued_config(explicit).rewards.terminal_energy_value_per_kwh == 0.7
    profiles = Profiles(np.full((2, 1), 4.), np.zeros(2), np.full(2, .35), np.zeros((2, 1)),
                        start_datetime=cfg.system.start_datetime,
                        resolution_minutes=cfg.system.resolution_minutes)
    outcomes = []
    for spend_initial in (False, True):
        env = MarketTrainingEnv(cfg, profiles)
        env.reset(42)
        records = []
        for period in range(2):
            mode = 2 if spend_initial and period == 0 else 0
            discharge = 4 / env.uc_bounds()[4] if mode == 2 else 0
            env.publish([mode, 1, 0, 0, 0, discharge, 0])
            records.append(env.step(consumer_actions(env, (0, 0, 0, 0))))
        assert not records[0].terminated and records[1].terminated
        assert records[0].metrics["terminal_energy_value"] == 0
        assert records[1].metrics["initial_energy_kwh"] == 5
        outcomes.append(records)
    idle, spend = outcomes
    # A one-period window reports a 0.33 reward advantage from initial inventory.
    assert spend[0].rewards["uc"] - idle[0].rewards["uc"] == pytest.approx(.33)
    assert spend[1].metrics["terminal_energy_value"] == pytest.approx(-.33)
    assert sum(r.rewards["uc"] for r in spend) == pytest.approx(sum(r.rewards["uc"] for r in idle))


def test_publish_has_two_phase_inputs_and_never_advances_time():
    env = MarketEnvFactory(config())(12)
    initial = env.reset(12)
    decision = env.publish([0, 0.7, 0.1, 0.5, 0.5, 0.3, 0.4])
    assert env.phase == Phase.CONSUMER_TURN
    assert env.engine.state.period_id == decision.period_id == 0
    np.testing.assert_array_equal(decision.uc_observation, initial)
    assert env.uc_value_dim == env.uc_observation_dim + 12
    np.testing.assert_array_equal(decision.uc_value_state[:-12], initial)
    np.testing.assert_array_equal(decision.uc_value_state[-12:], np.eye(12)[0])
    assert decision.uc_action.shape == (6,)
    assert set(decision.mapping_context) == {
        "ess_energy_kwh",
        "der_generation_kw",
        "grid_import_max_kw",
        "ess_energy_min_kwh",
        "ess_energy_capacity_kwh",
        "ess_charge_max_kw",
        "ess_discharge_max_kw",
        "ess_charge_efficiency",
        "ess_discharge_efficiency",
        "dt_hours",
        "dr_procurement_max_kw",
        "event_active",
    }
    assert decision.consumer_value_state.shape == (env.consumer_value_dim,)
    np.testing.assert_array_equal(
        decision.consumer_value_state,
        np.concatenate([initial, *decision.consumer_observations.values()]),
    )
    assert all(bound.shape == (13,) for bound in decision.consumer_bounds.values())
    result = env.step(consumer_actions(env))
    assert env.engine.state.period_id == 1
    assert env.phase == Phase.UC_TURN
    assert result.metrics["physical_periods"] == 1
    assert set(result.rewards) == {"uc", *env.consumer_ids}


@pytest.mark.parametrize("start,months", [("2026-01-31T23:45:00+08:00", (1, 2)),
                                        ("2026-12-31T23:45:00+08:00", (12, 1))])
def test_uc_value_public_calendar_follows_decision_time(start, months):
    base = config(steps=2)
    cfg = replace(base, system=replace(base.system, start_datetime=start))
    env = MarketEnvFactory(cfg)(12)
    env.reset(12)
    for month in months:
        actor_obs = env.engine.uc_observation().copy()
        decision = env.publish([0, .7, .1, .5, .5, .3, .4])
        np.testing.assert_array_equal(decision.uc_value_state[:-12], actor_obs)
        np.testing.assert_array_equal(decision.uc_value_state[-12:], np.eye(12)[month - 1])
        env.step(consumer_actions(env))


def test_consumer_value_state_changes_with_next_announcement():
    env = MarketEnvFactory(config())(12)
    env.reset(12)
    env.publish([0, 0.2, 0, 0, 0, 0, 0.2])
    env.step(consumer_actions(env))
    snapshot = env.get_state()
    first = env.publish([0, 0.2, 0, 0, 0, 0, 0.2])
    env.set_state(snapshot)
    second = env.publish([0, 0.8, 0, 0, 0, 0, 0.8])
    np.testing.assert_array_equal(first.uc_observation, second.uc_observation)
    assert not np.array_equal(first.consumer_value_state, second.consumer_value_state)


def test_pending_snapshot_restores_exact_market_transition():
    env = MarketEnvFactory(config())(12)
    env.reset(12)
    env.publish([0, 0.7, 0, 0.3, 0.4, 0.5, 0.6])
    snapshot = pickle.loads(pickle.dumps(env.get_state()))
    actions = consumer_actions(env)
    expected = env.step(actions)
    env.set_state(snapshot)
    actual = env.step(actions)
    assert actual.rewards == expected.rewards
    assert actual.metrics == expected.metrics
    np.testing.assert_array_equal(
        actual.next_uc_observation, expected.next_uc_observation
    )


def test_only_true_terminal_has_no_bootstrap_and_terminal_value_once():
    env = MarketEnvFactory(config(steps=2))(12)
    env.reset(12)
    for index in range(2):
        env.publish([0, 0.7, 0, 0, 0, 0, 0.4])
        result = env.step(consumer_actions(env))
        assert result.terminated == (index == 1)
    assert result.next_uc_observation is None
    assert not result.data_exhausted
    assert len(env.engine.records) == 2
    assert env.phase == Phase.TERMINATED


def test_continuous_exhaustion_is_not_terminal_and_can_append():
    cfg = config(mode="continuous", steps=2)
    env = MarketEnvFactory(cfg)(12)
    env.reset(12)
    for _ in range(2):
        env.publish([0, 0.7, 0, 0, 0, 0, 0.4])
        result = env.step(consumer_actions(env))
    assert not result.terminated
    assert result.data_exhausted
    assert result.next_uc_observation is None
    with pytest.raises(ProfileExhaustedError):
        env.publish([0, 0.7, 0, 0, 0, 0, 0.4])
    continuation = replace(
        cfg,
        system=replace(
            cfg.system, start_datetime=env.engine.state.timestamp.isoformat()
        ),
    )
    observation = env.append_profiles(scenario_profiles(continuation, 13, "train"))
    assert observation[0] == 2
    assert observation[1] == -1
    env.publish([0, 0.7, 0, 0, 0, 0, 0.4])
    assert env.step(consumer_actions(env)).next_uc_observation[0] == 3


def test_seed_reproduces_scenarios_and_different_seed_changes_inputs():
    factory = MarketEnvFactory(config(), split="validation")
    a, b, c = factory(12), factory(12), factory(13)
    assert a.engine.profiles.fingerprint() == b.engine.profiles.fingerprint()
    assert a.engine.profiles.fingerprint() != c.engine.profiles.fingerprint()
    assert a.engine.profiles.metadata["split"] == "validation"
    np.testing.assert_array_equal(a.reset(12), b.reset(12))


def test_due_queue_and_monthly_amount_remain_private_to_consumer_value():
    cfg = config()
    env = MarketTrainingEnv(cfg)
    consumers = (
        ConsumerState("consumer_0", 73, (ShiftTask(-1, 1, 0.25),)),
        ConsumerState("consumer_1", 7),
    )
    env.reset(12, options={"consumer_states": consumers})
    decision = env.publish([0, 0.7, 0, 0, 0, 0, 0.4])
    assert decision.consumer_observations["consumer_0"][5] == 73
    assert decision.consumer_observations["consumer_1"][5] == 7
    assert "monthly_uc_energy_kwh" not in env.uc_observation_fields
    assert all("consumer_" not in key for key in decision.mapping_context)


def test_raw_unclipped_controls_are_saved_separately_from_physical_actions():
    env = MarketEnvFactory(config())(12)
    env.reset(12)
    latent = np.array([2, 3, -2, 4, 0.5, 0.4, 3.0])
    decision = env.publish(latent)
    np.testing.assert_array_equal(decision.raw_controls, latent)
    assert decision.uc_action[0] == env.config.system.grid_import_max_kw
    assert decision.uc_action[1] == decision.uc_action[2] == decision.uc_action[3] == 0
    latent[1] = 999
    assert decision.raw_controls[1] == 3


def test_metrics_use_single_physical_period_units():
    env = MarketEnvFactory(config())(12)
    env.reset(12)
    env.publish([0, 0.7, 0, 0, 0, 0, 0.4])
    result = env.step(consumer_actions(env))
    record = env.engine.latest_record
    assert result.metrics["uc_cash"] == record.ledger.cash_profit
    assert result.metrics["served_kwh"] == record.physical.supply_served_kw * 0.25
    assert result.metrics["consumer_cost"] == sum(
        x.total_cost for x in record.ledger.consumers.values()
    )


def test_factory_contract_is_json_stable_and_binds_configuration_and_splits():
    cfg = config()
    splits = {"train": (1, 2), "validation": (11, 12), "test": (21, 22)}
    factory = MarketEnvFactory(cfg, seed_splits=splits)
    contract = factory.get_contract()
    assert contract == json.loads(json.dumps(contract))
    assert contract == MarketEnvFactory(cfg, seed_splits=splits).get_contract()
    assert len(contract["profile_rule_sha256"]) == 64
    assert contract["seed_splits"]["train"] == [1, 2]
    assert contract != replace(factory, split="evaluation").get_contract()
    assert contract != replace(factory, profile_periods=16).get_contract()
    assert (
        contract
        != replace(factory, seed_splits={**splits, "train": (3, 4)}).get_contract()
    )
    changed = replace(cfg, rewards=replace(cfg.rewards, adjustment_rate_per_kwh=0.75))
    assert contract != replace(factory, config=changed).get_contract()
    contract["seed_splits"]["train"].append(999)
    assert factory.get_contract()["seed_splits"]["train"] == [1, 2]
