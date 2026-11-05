import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from v9_checks import compare_record

from snmarket import ConsumerAction as C
from snmarket import ConsumerState, Phase, Profiles, ShiftTask, StageError
from snmarket import UCAction as U
from snmarket.engine import ProfileExhaustedError
from snmarket.observation import consumer_fields, uc_fields


def actions(engine, **kw):
    return {identity: C(curtail_kw=kw.get("cut", 0), signed_shift_kw=kw.get("shift", 0), der_preference_fraction=kw.get("rho", 0), der_bid_per_kwh=kw.get("bid", 0.3))
            for identity in engine.consumer_ids}


def test_two_stage_clock_atomic_candidate_and_idempotent_concurrent_commit(make_engine):
    engine = make_engine([[4], [4]])
    obs = engine.reset()
    assert len(obs) == engine.config.market.shift_deadline_steps + 20
    before = engine.state
    uc = U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0)
    engine.apply_uc_action(uc)
    assert engine.state == before
    assert engine.phase == Phase.CONSUMER_TURN
    delta = engine.evaluate_consumer_actions(actions(engine), transition_id="once")
    assert engine.state == before
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(engine.commit, [delta, copy.deepcopy(delta)]))
    assert records[0] == records[1]
    assert engine.state.period_id == 1
    assert len(engine.records) == 1
    assert engine.run_period(uc, actions(engine), transition_id="once") == records[0]
    assert engine.state.period_id == 1
    with pytest.raises(ValueError):
        engine.run_period(U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), actions(engine), transition_id="once")


@pytest.mark.parametrize("bad", [[0] * 5, [0] * 7, [-1, 0, 0, 0, 0, 0],
                                  [101, 0, 0, 0, 0, 0], [4, 0, 0, 5, 0, 0],
                                  [4, 9, 0, 0, 0, 0], [4, 0, 1, 0, 1, 0],
                                  [4, 0, 0, 0, 30, 0], [4, 0, 0, 0, 0, 41],
                                  [float("nan"), 0, 0, 0, 0, 0],
                                  [4, 0, 0, float("inf"), 0, 0]])
def test_invalid_uc_never_changes_state_or_announces(make_engine, bad):
    engine = make_engine([4], der=8)
    engine.reset()
    before = engine.get_state()
    with pytest.raises(ValueError):
        engine.apply_uc_action(bad)
    assert engine.get_state() == before


@pytest.mark.parametrize("bad", [[0, 0, 0], [0, 0, 0, 0.3, 0], [-1, 0, 0, 0.3],
                                  [5, 0, 0, 0.3], [0, -1, 0, 0.3], [0, 1, 0, 0.3],
                                  [0, 0, 1.1, 0.3], [0, 0, 0, 0.2], [0, 0, 0, 0.5],
                                  [0, 0, 0, float("nan")]])
def test_invalid_consumer_preserves_pending_announcement(make_engine, bad):
    engine = make_engine([4], flex=4)
    engine.reset()
    engine.apply_uc_action(U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    before = engine.get_state()
    with pytest.raises(ValueError):
        engine.apply_consumer_actions({"consumer_0": bad})
    assert engine.get_state() == before


def test_wrong_agent_sets_and_stages_rejected(make_engine):
    engine = make_engine([4, 4])
    with pytest.raises(StageError):
        engine.apply_uc_action(U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    engine.reset()
    with pytest.raises(StageError):
        engine.apply_consumer_actions(actions(engine))
    engine.apply_uc_action(U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    for bad in ({}, {"consumer_0": C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)}, {**actions(engine), "other": C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)}):
        with pytest.raises(ValueError):
            engine.apply_consumer_actions(bad)
    with pytest.raises(StageError):
        engine.apply_uc_action(U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))


def test_injected_failure_and_tampered_candidate_preserve_state(make_engine, monkeypatch):
    import snmarket.engine as module
    engine = make_engine([[4], [4]])
    engine.reset()
    engine.apply_uc_action(U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    before = engine.get_state()
    original = module.settle
    fault = RuntimeError("injected settlement fault")

    def fail(*args, **kwargs):
        raise fault

    monkeypatch.setattr(module, "settle", fail)
    with pytest.raises(RuntimeError, match="injected") as error:
        engine.apply_consumer_actions(actions(engine))
    assert error.value is fault
    assert engine.get_state() == before
    monkeypatch.setattr(module, "settle", original)
    delta = engine.evaluate_consumer_actions(actions(engine))
    with pytest.raises(ValueError, match="modified"):
        engine.commit(replace(delta, after=replace(delta.after, ess_energy_kwh=9)))
    assert engine.state.period_id == 0
    engine.commit(delta)


def test_snapshot_restore_both_phases_and_public_objects_do_not_alias(make_engine):
    engine = make_engine([[4], [4]])
    engine.reset(seed=7)
    initial = engine.get_state()
    uc = U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0)
    engine.apply_uc_action(uc)
    pending = engine.get_state()
    first = engine.apply_consumer_actions(actions(engine), transition_id="same")
    settled = engine.get_state()
    assert first.ledger.cash_profit < 0
    engine.set_state(pending)
    assert engine.apply_consumer_actions(actions(engine), transition_id="same") == first
    engine.set_state(initial)
    assert engine.run_period(uc, actions(engine), transition_id="same") == first
    engine.set_state(settled)
    first.consumer_actions.clear()
    assert engine.latest_record.consumer_actions
    exposed = engine.records
    exposed[0].consumer_actions.clear()
    assert len(engine.records) == 1
    assert engine.records[0].consumer_actions


def test_scalar_copy_keeps_public_records_detached(make_engine):
    engine = make_engine([[4], [4]])
    engine.reset()
    record = engine.run_period(
        U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0,
          grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0),
        actions(engine),
    )
    before = engine.get_state()
    scalars = (
        record.uc_action, record.consumer_actions["consumer_0"],
        ShiftTask(0, 1, 0.5), record.local_bounds["consumer_0"],
        record.market.allocations["consumer_0"].necessary,
        record.physical, record.dr.consumers["consumer_0"],
        record.ledger.consumers["consumer_0"],
    )
    for item in scalars:
        cloned = copy.deepcopy(item)
        assert cloned == item and cloned is not item
        field = next(iter(vars(item)))
        original = getattr(item, field)
        # Even bypassing frozen dataclass assignment must not corrupt the source.
        object.__setattr__(cloned, field, None)
        assert getattr(item, field) == original
    assert engine.get_state() == before


@pytest.mark.parametrize("kind", ["soc", "phase", "offer", "payments", "history", "feedback", "initial_energy"])
def test_snapshot_corruption_rejected_without_mutation(make_engine, kind):
    engine = make_engine([[4], [4]])
    engine.reset()
    engine.run_period(U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), actions(engine), transition_id="one")
    before = engine.get_state()
    bad = {"soc": replace(before, state=replace(before.state, ess_energy_kwh=9)),
           "phase": replace(before, phase=Phase.TERMINATED),
           "offer": replace(before, pending_plan=U(grid_import_offer_kw=1, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0)),
           "payments": replace(before, payment_keys=frozenset()),
           "history": replace(before, records={}),
           "feedback": replace(before, state=replace(before.state, last_totals=(0.0,) * 6)),
           "initial_energy": replace(before, state=replace(before.state, initial_ess_energy_kwh=2))}[kind]
    with pytest.raises(ValueError):
        engine.set_state(bad)
    assert engine.get_state() == before


def test_month_transition_freezes_price_then_resets_next_month(make_engine):
    engine = make_engine([[8], [4]], start="2026-01-31T23:45:00+08:00")
    consumer = ConsumerState("consumer_0", monthly_uc_energy_kwh=99)
    engine.reset(options=dict(consumer_states=(consumer,)))
    before = engine.state
    record = engine.run_period(U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), actions(engine))
    compare_record(engine, before, record)
    assert record.ledger.consumers["consumer_0"].energy_payment == pytest.approx(0.8)
    assert record.monthly_closing_energy_kwh["consumer_0"] == 101
    assert record.month_rolled
    assert engine.state.consumers[0].monthly_uc_energy_kwh == 0
    assert engine.state.timestamp.month == 2


def test_observation_dimensions_order_aggregate_due_and_no_private_uc_fields(make_engine):
    engine = make_engine([[4, 0], [4, 0]], flex=[0, 4], deadline=4)
    consumers = (ConsumerState("consumer_0", tasks=(ShiftTask(-4, 0, 1),)), ConsumerState("consumer_1"))
    observation = engine.reset(options=dict(consumer_states=consumers))
    names = uc_fields(4)
    uc = dict(zip(names, observation))
    assert len(observation) == 24
    assert uc["aggregate_due_kw"] == 0
    assert uc["aggregate_due_bucket_0_kwh"] == 1
    assert uc["aggregate_baseline_kw"] == 4
    assert uc["weighted_uc_tariff"] == pytest.approx(0.4)
    assert all(uc[name] == 0 for name in names[-6:])
    assert all("consumer_" not in name for name in names)
    obs = engine.apply_uc_action(U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    for i in engine.consumer_ids:
        assert len(obs[i]) == 28
        assert np.isfinite(obs[i]).all()
    assert dict(zip(consumer_fields(4), obs["consumer_0"]))["due_bucket_0_kwh"] == 1


def test_continuous_exhaustion_append_keeps_soc_queues_clock_and_no_terminal(make_engine):
    engine = make_engine([4], flex=4, mode="continuous", deadline=1)
    observation = engine.reset()
    assert observation[1] == -1
    record = engine.run_period(U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), actions(engine, shift=4))
    assert not record.terminated
    assert engine.state.consumers[0].backlog_kwh == 1
    assert record.ledger.terminal_pending_cost == 0
    assert engine.data_exhausted
    before = engine.get_state()
    with pytest.raises(ProfileExhaustedError):
        engine.apply_uc_action(U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    assert engine.get_state() == before
    appended = Profiles([[4]], [0], [0.5], [[4]], rigid_kw=[[0]],
                        start_datetime=engine.state.timestamp.isoformat(), resolution_minutes=15)
    engine.append_profiles(appended)
    assert engine.state == before.state
    engine.run_period(U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), actions(engine, shift=-4))
    assert engine.state.period_id == 2
    assert engine.state.consumers[0].backlog_kwh == 0
    assert engine.phase == Phase.UC_TURN


def test_heterogeneous_coefficients_and_queue_caps_reach_individual_ledgers(make_engine):
    engine = make_engine([4, 4], flex=4,
                         market=dict(shift_backlog_limit_kwh=(2, 5), der_bid_cap=(0.35, 0.4)),
                         rewards=dict(actual_compensation_per_kwh=(1, 2),
                                      consumer_native_loss_rate_per_kwh=(3, 4),
                                      adjustment_rate_per_kwh=(0.5, 0.75)))
    engine.reset()
    before = engine.state
    record = engine.run_period(U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0),
                               {"consumer_0": C(curtail_kw=1, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.35), "consumer_1": C(curtail_kw=2, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.4)})
    compare_record(engine, before, record)
    assert record.ledger.consumers["consumer_0"].compensation_income == 0.75
    assert record.ledger.consumers["consumer_1"].compensation_income == 1.0


@pytest.mark.parametrize("energy", [0.1038246753129, 0.01, 0.3, 0.7, 1.0, 5.0])
def test_exact_full_discharge_reaches_valid_zero_soc(make_engine, energy):
    engine = make_engine([[100], [100]], ess=energy)
    engine.reset()
    discharge = 0.95 * energy / 0.25
    record = engine.run_period(U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=discharge, dr_procurement_cap_kw=0), actions(engine))
    assert record.physical.ess_energy_after_kwh >= 0
    assert engine.state.ess_energy_kwh == pytest.approx(0, abs=1e-14)
    assert np.isfinite(engine.uc_observation()).all()


def test_final_period_rejects_late_new_shift_and_settles_terminal_once(make_engine):
    engine = make_engine([4], flex=4,
                         rewards=dict(consumer_terminal_pending_rate_per_kwh=3,
                                      uc_terminal_pending_rate_per_kwh=2))
    c = ConsumerState("consumer_0", tasks=(ShiftTask(-2, 2, 1),))
    engine.reset(options=dict(consumer_states=(c,)))
    engine.apply_uc_action(U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    with pytest.raises(ValueError):
        engine.apply_consumer_actions(actions(engine, shift=1))
    record = engine.apply_consumer_actions(actions(engine), transition_id="terminal")
    assert record.terminated
    assert record.ledger.terminal_pending_cost == 2
    assert record.ledger.consumers["consumer_0"].terminal_pending_cost == 3
    assert engine.apply_consumer_actions(actions(engine), transition_id="terminal") == record
    assert len(engine.records) == 1
