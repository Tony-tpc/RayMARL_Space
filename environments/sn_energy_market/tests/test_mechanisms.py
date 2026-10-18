from dataclasses import astuple

import numpy as np
import pytest
from v9_checks import compare_record

from snmarket import ConsumerAction as C
from snmarket import ConsumerState, ShiftTask
from snmarket import UCAction as U
from snmarket.action import build_local_bounds, map_consumer_controls, map_uc_controls
from snmarket.uc import validate_uc_action


def run(engine, uc, actions, options=None):
    engine.reset(options=options)
    before = engine.state
    actions = dict(zip(engine.consumer_ids, actions))
    record = engine.run_period(uc, actions)
    compare_record(engine, before, record)
    return record


@pytest.mark.parametrize("uc", [U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), U(grid_import_offer_kw=4, der_planned_curtailment_kw=1, der_to_ess_kw=1, grid_to_ess_kw=2, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0),
                                U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=2, dr_procurement_cap_kw=0), U(grid_import_offer_kw=0, der_planned_curtailment_kw=4, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0),
                                U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=4, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=4, ess_discharge_offer_kw=0, dr_procurement_cap_kw=3)])
def test_six_actions_preserved_with_full_oracle(make_engine, uc):
    engine = make_engine([[6, 5], [6, 5]], der=4, flex=4)
    record = run(engine, uc, [C(curtail_kw=1, signed_shift_kw=0, der_preference_fraction=0.5, der_bid_per_kwh=0.4), C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=1, der_bid_per_kwh=0.3)])
    assert record.uc_action == uc
    assert record.physical.der_to_ess_kw == uc.der_to_ess_kw
    assert record.physical.grid_to_ess_kw == uc.grid_to_ess_kw
    assert record.physical.der_planned_curtailment_kw == uc.der_planned_curtailment_kw
    assert record.physical.grid_buy_kw <= uc.grid_import_offer_kw + 1e-9
    assert record.physical.ess_discharge_kw <= uc.ess_discharge_offer_kw + 1e-9


def test_zero_sale_keeps_firm_charging_and_curtailment(make_engine):
    record = run(make_engine([0], der=8), U(grid_import_offer_kw=3, der_planned_curtailment_kw=2, der_to_ess_kw=2, grid_to_ess_kw=3, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    assert record.physical.grid_buy_kw == 3
    assert record.physical.der_to_ess_kw == 2
    assert record.physical.der_spill_kw == 6
    assert record.physical.der_unsold_kw == 4
    assert record.physical.ess_energy_after_kwh == pytest.approx(5 + 0.95 * 5 * 0.25)


def test_equal_bids_proportional_order_independent(make_engine):
    actions = [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=1, der_bid_per_kwh=0.3), C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=1, der_bid_per_kwh=0.3)]
    a = run(make_engine([4, 2], der=3), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), actions)
    b = run(make_engine([2, 4], der=3), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), actions)
    assert [a.market.allocations[i].der_kw for i in a.market.allocations] == [2, 1]
    assert [b.market.allocations[i].der_kw for i in b.market.allocations] == [1, 2]


def test_higher_der_bid_paid_at_own_bid_and_uc_at_fixed_tariff(make_engine):
    engine = make_engine([4, 4], der=4, market=dict(uc_flat_rates=(0.9, 0.9, 0.9)))
    result = run(engine, U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=1, der_bid_per_kwh=0.7), C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=1, der_bid_per_kwh=0.4)])
    assert result.market.allocations["consumer_0"].der_kw == 4
    assert result.ledger.consumers["consumer_0"].energy_payment == pytest.approx(0.7)
    assert result.ledger.consumers["consumer_1"].energy_payment == pytest.approx(0.9)


def test_der_rescue_uses_original_bid_and_only_remaining_capacity(make_engine):
    result = run(make_engine([4, 4], der=5), U(grid_import_offer_kw=3, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0),
                 [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.4), C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    assert result.market.allocations["consumer_0"].necessary.der_rescue_kw == 2.5
    assert result.market.allocations["consumer_1"].necessary.der_rescue_kw == 2.5
    assert result.ledger.consumers["consumer_0"].der_payment == pytest.approx(0.25)


def test_necessary_layer_beats_high_bid_early_repayment(make_engine):
    engine = make_engine([4, 0], der=4, flex=[0, 4])
    states = (ConsumerState("consumer_0"), ConsumerState("consumer_1", tasks=(ShiftTask(-1, 3, 1),)))
    result = run(engine, U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=1, der_bid_per_kwh=0.3), C(curtail_kw=0, signed_shift_kw=-4, der_preference_fraction=1, der_bid_per_kwh=0.4)],
                 dict(consumer_states=states))
    assert result.market.allocations["consumer_0"].der_kw == 4
    assert result.services["consumer_1"].repay_actual_kw == 0
    assert result.services["consumer_1"].backlog_after_kwh == 1


def test_actual_repay_fifo_earliest_deadline_and_native_priority(make_engine):
    engine = make_engine([[2], [2]], flex=8)
    tasks = (ShiftTask(-3, 1, 0.5), ShiftTask(-2, 0, 0.5), ShiftTask(-1, 2, 0.5))
    engine.reset(options=dict(consumer_states=(ConsumerState("consumer_0", tasks=tasks),)))
    before = engine.state
    normalized = before.consumers[0].tasks
    result = engine.run_period(U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), {"consumer_0": C(curtail_kw=0, signed_shift_kw=-4, der_preference_fraction=0, der_bid_per_kwh=0.3)})
    compare_record(engine, before, result)
    service = result.services["consumer_0"]
    assert service.native_served_kw == 2
    assert service.repay_actual_kw == 2
    assert service.tasks_after == (normalized[0], normalized[2])
    assert service.overdue_after_kwh == 0


def test_due_local_capacity_and_terminal_no_deferral(make_engine):
    engine = make_engine([4], flex=4)
    state = ConsumerState("consumer_0", tasks=(ShiftTask(-4, 0, 2),))
    engine.reset(options=dict(consumer_states=(state,)))
    bounds = build_local_bounds(state, 4, 0, 4, 0, engine.config, timestamp=engine.state.timestamp)
    assert bounds.cut_max_kw == 0
    assert bounds.due_kw == 4
    assert bounds.shift_interval(0) == (-4, -4)
    before = engine.state
    record = engine.run_period(U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), {"consumer_0": C(curtail_kw=0, signed_shift_kw=-4, der_preference_fraction=0, der_bid_per_kwh=0.3)})
    compare_record(engine, before, record)
    assert record.services["consumer_0"].due_capacity_shortfall_kwh == 1
    assert record.services["consumer_0"].overdue_after_kwh == 1


def test_output_map_covers_target_actions_and_zero_domains(make_engine):
    engine = make_engine([0], der=8, grid=10, flex=0)
    engine.reset()
    for z, expected in [([0.4, 0.25, 1 / 3, 0.5, 1, 0.1], U(grid_import_offer_kw=4, der_planned_curtailment_kw=2, der_to_ess_kw=2, grid_to_ess_kw=2, ess_discharge_offer_kw=0, dr_procurement_cap_kw=4)),
                        ([0.4, 0.25, 0, 0, -0.25, 0.1], U(grid_import_offer_kw=4, der_planned_curtailment_kw=2, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=4.75, dr_procurement_cap_kw=4)),
                        ([0, 1, 0, 0, 0, 0], U(grid_import_offer_kw=0, der_planned_curtailment_kw=8, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))]:
        action = map_uc_controls(z, engine.state, 8, engine.config)
        assert astuple(action) == pytest.approx(astuple(expected))
        validate_uc_action(action, engine.state, 8, engine.config)
    zero = make_engine([0], der=0, grid=0, flex=0, ess=0, active=False)
    zero.reset()
    assert astuple(map_uc_controls([10] * 6, zero.state, 0, zero.config)) == (0,) * 6


def test_random_policy_output_is_feasible_but_never_applied_implicitly(make_engine):
    engine = make_engine([4], der=8, grid=10, flex=4)
    engine.reset()
    bounds = build_local_bounds(engine.state.consumers[0], 4, 0, 4, 0,
                                engine.config, timestamp=engine.state.timestamp)
    rng = np.random.default_rng(713)
    for _ in range(1000):
        action = map_uc_controls(rng.normal(size=6), engine.state, 8, engine.config)
        validate_uc_action(action, engine.state, 8, engine.config)
        consumer = map_consumer_controls(rng.uniform(size=4), bounds)
        low, high = bounds.shift_interval(consumer.curtail_kw)
        assert low - 1e-10 <= consumer.signed_shift_kw <= high + 1e-10
        assert consumer.der_bid_per_kwh <= bounds.der_bid_max
    before = engine.get_state()
    with pytest.raises(ValueError):
        engine.apply_uc_action([-0.5] * 6)
    assert engine.get_state() == before


@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_real_tiny_task_is_not_erased_by_absolute_tolerance(make_engine, fraction):
    energy = 1e-13
    engine = make_engine([0], flex=1)
    initial = ConsumerState("consumer_0", tasks=(ShiftTask(-1, 3, energy),))
    power = fraction * energy / 0.25
    record = run(engine, U(grid_import_offer_kw=power, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=0, signed_shift_kw=-power, der_preference_fraction=0, der_bid_per_kwh=0.3)],
                 dict(consumer_states=(initial,)))
    queue = record.services["consumer_0"].tasks_after
    assert len(queue) == 1
    assert queue[0].remaining_kwh == pytest.approx(energy * (1 - fraction), abs=1e-28, rel=1e-12)
