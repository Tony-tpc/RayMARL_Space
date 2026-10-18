"""Economic outcomes specified by v9, including the explicitly retained issues."""

import pytest
from v9_checks import compare_record

from snmarket import ConsumerAction as C
from snmarket import ConsumerState, ShiftTask
from snmarket import UCAction as U


def settle(engine, uc, actions, states=None):
    engine.reset(options={} if states is None else {"consumer_states": states})
    before = engine.state
    record = engine.run_period(uc, dict(zip(engine.consumer_ids, actions)))
    compare_record(engine, before, record)
    return record


def test_u03_normal_supply_beats_unpaid_induced_cut_in_document_case(make_engine):
    args = dict(flex=4, grid_cost=0.65, rewards=dict(adjustment_rate_per_kwh=0.75))
    full = settle(make_engine([4], **args), U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    unpaid = settle(make_engine([4], **args), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=4, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    paid = settle(make_engine([4], **args), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=4), [C(curtail_kw=4, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    assert full.rewards["uc"] == pytest.approx(-0.25)
    assert unpaid.rewards["uc"] == pytest.approx(-2)
    assert paid.rewards["uc"] == pytest.approx(-0.5)
    assert full.rewards["uc"] > unpaid.rewards["uc"]
    assert unpaid.rewards["consumer_0"] == pytest.approx(-0.75)
    assert paid.rewards["consumer_0"] == pytest.approx(-0.25)


def test_a01_purchase_credit_does_not_claim_incremental_dr(make_engine):
    base = settle(make_engine([4], flex=4), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=4, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    paid = settle(make_engine([4], flex=4), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=4), [C(curtail_kw=4, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    assert base.dr.net_dr_kw == paid.dr.net_dr_kw == 4
    assert base.physical.grid_buy_kw == paid.physical.grid_buy_kw == 0
    assert paid.ledger.cash_profit - base.ledger.cash_profit == pytest.approx(-0.5)
    assert paid.rewards["uc"] - base.rewards["uc"] == pytest.approx(1.5)


def test_a02_extra_due_reservation_remains_visible(make_engine):
    state = ConsumerState("consumer_0", tasks=(ShiftTask(-2, 0, 0.5),))
    record = settle(make_engine([4], flex=6), U(grid_import_offer_kw=2, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=2),
                    [C(curtail_kw=4, signed_shift_kw=-2, der_preference_fraction=0, der_bid_per_kwh=0.3)], (state,))
    assert record.services["consumer_0"].backlog_after_kwh == 0
    assert record.services["consumer_0"].native_lost_kw == 0
    assert record.dr.net_dr_kw == record.dr.procured_kw == 2
    assert record.ledger.supply_reference_kw == 6
    assert record.ledger.supply_shortfall_kw == 2
    assert record.ledger.supply_shortfall_cost == 1


def test_u02_der_to_uc_substitution_is_an_economic_result(make_engine):
    args = dict(der=4, grid_cost=0.2, market=dict(uc_flat_rates=(0.9, 0.9, 0.9)))
    direct = settle(make_engine([4], **args), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=1, der_bid_per_kwh=0.3)])
    switch = settle(make_engine([4], **args), U(grid_import_offer_kw=4, der_planned_curtailment_kw=4, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=1, der_bid_per_kwh=0.3)])
    assert direct.rewards["uc"] == pytest.approx(0.3)
    assert switch.rewards["uc"] == pytest.approx(0.6)
    assert switch.ledger.consumers["consumer_0"].energy_payment == pytest.approx(0.9)
    assert switch.services["consumer_0"].native_lost_kw == 0


def test_a03_unused_announced_capacity_plateau(make_engine):
    a = settle(make_engine([4], flex=4), U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=2, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    b = settle(make_engine([4], flex=4), U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=2, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    assert a.physical.grid_buy_kw == b.physical.grid_buy_kw == 2
    assert a.rewards == b.rewards
    assert a.physical.grid_offer_unused_kw == 2
    assert b.physical.grid_offer_unused_kw == 6


@pytest.mark.parametrize("price,active,cap,credit", [(0.0, True, 4, 0), (0.5, False, 0, 0),
                                                    (0.5, True, 4, 4)])
def test_u03_credit_requires_event_positive_price_and_actual_dr(make_engine, price, active, cap, credit):
    engine = make_engine([4], flex=4, active=active, market=dict(dr_incentive_per_kwh=price))
    result = settle(engine, U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=cap), [C(curtail_kw=4, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    assert result.dr.supply_credit_kw == credit
    assert result.ledger.supply_shortfall_kw == 4 - credit
    if active and price > 0:
        no_response = settle(make_engine([4], flex=4), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=cap), [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
        assert no_response.dr.supply_credit_kw == 0


def test_net_dr_cancels_repayment_request_even_without_service(make_engine):
    states = (ConsumerState("consumer_0"), ConsumerState("consumer_1", tasks=(ShiftTask(-1, 2, 1),)))
    result = settle(make_engine([4, 0], flex=[4, 4]), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=8),
                    [C(curtail_kw=4, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3), C(curtail_kw=0, signed_shift_kw=-4, der_preference_fraction=0, der_bid_per_kwh=0.3)], states)
    assert result.dr.positive_dr_kw == 4
    assert result.dr.net_dr_kw == result.dr.procured_kw == 0
    assert result.services["consumer_1"].backlog_after_kwh == 1
    assert result.ledger.dr_payment == 0


def test_positive_dr_value_can_reopen_low_coefficient_incentive(make_engine):
    args = dict(flex=4, grid_cost=0.65, rewards=dict(uc_supply_shortfall_rate_per_kwh=0.5,
                                                  dr_value_rate_per_kwh=0.4))
    full = settle(make_engine([4], **args), U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    cut = settle(make_engine([4], **args), U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=4, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)])
    assert full.rewards["uc"] == pytest.approx(-0.25)
    assert cut.rewards["uc"] == pytest.approx(-0.1)


@pytest.mark.parametrize("minutes", [5, 15, 30, 60])
def test_cost_units_per_energy_per_time_and_per_assessment(make_engine, minutes):
    state = ConsumerState("consumer_0", tasks=(ShiftTask(-2, 0, 1),))
    rewards = dict(uc_due_rate_per_kwh=3.0, consumer_wait_rate_per_kwh_hour=1.0,
                   consumer_overdue_rate_per_kwh_hour=2.0, consumer_terminal_pending_rate_per_kwh=5,
                   uc_terminal_pending_rate_per_kwh=7, terminal_energy_value_per_kwh=0.2,
                   ess_throughput_rate_per_kwh=0.3)
    record = settle(make_engine([0], flex=0, ess=0, dt_minutes=minutes, rewards=rewards),
                    U(grid_import_offer_kw=2, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=2, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), [C(curtail_kw=0, signed_shift_kw=0, der_preference_fraction=0, der_bid_per_kwh=0.3)], (state,))
    dt = minutes / 60
    consumer = record.ledger.consumers["consumer_0"]
    assert record.ledger.due_cost == 3
    assert consumer.waiting_cost == dt
    assert consumer.overdue_cost == 2 * dt
    assert consumer.terminal_pending_cost == 5
    assert record.ledger.terminal_pending_cost == 7
    assert record.ledger.ess_cost == pytest.approx(0.6 * dt)
    assert record.ledger.terminal_energy_value == pytest.approx(0.2 * 0.95 * 2 * dt)


def test_document_full_worked_example_end_to_end(make_engine):
    engine = make_engine([[6, 5, 4]] * 8, der=6, grid=8, flex=[2, 1, 1],
                         grid_cost=0.65, deadline=4,
                         market=dict(uc_flat_rates=(0.9, 0.9, 0.9), dr_incentive_per_kwh=0.2),
                         rewards=dict(dr_value_rate_per_kwh=0.4))
    states = (ConsumerState("consumer_0"), ConsumerState("consumer_1"),
              ConsumerState("consumer_2", tasks=(ShiftTask(-1, 3, 0.25),)))
    record = settle(engine, U(grid_import_offer_kw=6, der_planned_curtailment_kw=1, der_to_ess_kw=1, grid_to_ess_kw=1, ess_discharge_offer_kw=0, dr_procurement_cap_kw=2),
                    [C(curtail_kw=1, signed_shift_kw=1, der_preference_fraction=0.5, der_bid_per_kwh=0.7), C(curtail_kw=1, signed_shift_kw=0, der_preference_fraction=0.5, der_bid_per_kwh=0.65), C(curtail_kw=0, signed_shift_kw=-1, der_preference_fraction=0.4, der_bid_per_kwh=0.6)], states)
    assert record.ledger.cash_profit == pytest.approx(-0.025)
    assert record.ledger.book_profit == pytest.approx(-0.06)
    assert record.ledger.uc_reward == pytest.approx(-1.86)
    assert [record.rewards[i] for i in engine.consumer_ids] == pytest.approx([-0.8, -0.7854166666667, -0.94375])
    assert record.physical.ess_energy_after_kwh == pytest.approx(5.475)
