"""Cross-period energy, current U01 rewards, and aggregate information equivalence."""
from dataclasses import fields, replace

import numpy as np
import pytest

from snmarket import ConsumerState, EnergyMarketEngine, Profiles, ShiftTask
from snmarket import UCAction as U


def test_shift_then_full_repayment_preserves_total_energy(make_engine):
    engine = make_engine([[4], [4]], flex=4, deadline=1)
    engine.reset()
    first = engine.run_period(U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=4), {"consumer_0": [0, 4, 0, .3]})
    second = engine.run_period(U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=4), {"consumer_0": [0, -4, 0, .3]})
    assert first.dr.net_dr_kw == 4
    assert second.dr.consumers["consumer_0"].signed_dr_kw == -4
    assert (first.physical.supply_served_kw + second.physical.supply_served_kw) * .25 == 2
    assert engine.state.consumers[0].backlog_kwh == 0
    assert first.services["consumer_0"].native_lost_kw == second.services["consumer_0"].native_lost_kw == 0


def test_current_u01_reward_includes_adequacy_beyond_historical_cash(make_engine):
    results = []
    for supply in (4, 0):
        engine = make_engine([4], grid_cost=1.7)
        engine.reset()
        results.append(engine.run_period(U(grid_import_offer_kw=supply, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), {"consumer_0": [0, 0, 0, .3]}))
    full, lost = results
    assert full.ledger.cash_profit == pytest.approx(-1.3)
    assert lost.ledger.cash_profit == pytest.approx(-1)
    assert full.rewards["uc"] == pytest.approx(-1.3)
    assert lost.rewards["uc"] == pytest.approx(-3)


def test_private_queue_redistribution_preserves_uc_aggregates_and_other_consumer_input(make_engine):
    a, b = (make_engine([[4, 4, 4]] * 4, flex=[4, 4, 4], deadline=4) for _ in range(2))
    initial = (ConsumerState("consumer_0"),
               ConsumerState("consumer_1", tasks=(ShiftTask(-2, 0, .25),)),
               ConsumerState("consumer_2", tasks=(ShiftTask(-2, 0, .5),)))
    changed = (initial[0], replace(initial[1], tasks=initial[2].tasks),
               replace(initial[2], tasks=initial[1].tasks))
    np.testing.assert_array_equal(a.reset(options={"consumer_states": initial}),
                                  b.reset(options={"consumer_states": changed}))
    oa = a.apply_uc_action(U(grid_import_offer_kw=20, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    ob = b.apply_uc_action(U(grid_import_offer_kw=20, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    np.testing.assert_array_equal(oa["consumer_0"], ob["consumer_0"])
    assert not np.array_equal(oa["consumer_1"], ob["consumer_1"])


def test_restored_history_order_is_chronological(make_engine):
    engine = make_engine([[4], [4]])
    engine.reset()
    first = engine.run_period(U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), {"consumer_0": [0, 0, 0, .3]})
    last = engine.run_period(U(grid_import_offer_kw=8, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), {"consumer_0": [0, 0, 0, .3]})
    snapshot = engine.get_state()
    reordered = replace(snapshot, records={last.transition_id: last, first.transition_id: first})
    engine.set_state(reordered)
    assert engine.latest_record == last
    assert [record.period_id for record in engine.records] == [0, 1]
    assert engine.consumer_observations(terminal=True)["consumer_0"][15] == 8


@pytest.mark.parametrize("delay,mode", [(4, "finite"), (96, "finite"), (24, "continuous")])
def test_explicit_horizon_delay_mode_combinations(make_engine, delay, mode):
    engine = make_engine([[4]] * 96, flex=4, deadline=delay, mode=mode)
    engine.reset()
    for period in range(96):
        engine.apply_uc_action(U(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
        bound = engine.local_bounds()["consumer_0"]
        assert bound.allow_new_shift == (mode == "continuous" or period + delay <= 95)
        low, high = bound.shift_interval(0)
        assert low == 0
        assert high == (4 if bound.allow_new_shift else 0)
        engine.apply_consumer_actions({"consumer_0": [0, 0, 0, .3]})


def test_consistent_power_energy_and_unit_coefficient_rescaling(make_engine):
    scale = 1000.0
    original = make_engine([[6, 5, 4]] * 8, der=6, grid=8, flex=[2, 1, 1], grid_cost=.65,
                           market={"uc_flat_rates": (.9, .9, .9), "dr_incentive_per_kwh": .2},
                           rewards={"dr_value_rate_per_kwh": .4, "der_operating_rate_per_kwh": .03})
    c, p = original.config, original.profiles
    physical_fields = ("grid_import_max_kw", "ess_energy_capacity_kwh", "ess_energy_min_kwh",
                       "ess_initial_energy_kwh", "ess_charge_max_kw", "ess_discharge_max_kw")
    system = replace(c.system, **{key: getattr(c.system, key) * scale for key in physical_fields})
    price_fields = ("uc_flat_rates", "uc_high_rates", "uc_peak_rates")
    market = replace(c.market, der_bid_cap=c.market.der_bid_cap / scale,
                     der_reference_rate_per_kwh=c.market.der_reference_rate_per_kwh / scale,
                     dr_incentive_per_kwh=c.market.dr_incentive_per_kwh / scale,
                     dr_procurement_max_kw=c.market.dr_procurement_max_kw * scale,
                     shift_backlog_limit_kwh=c.market.shift_backlog_limit_kwh * scale,
                     tier_thresholds_kwh=tuple(v * scale for v in c.market.tier_thresholds_kwh),
                     **{key: tuple(v / scale for v in getattr(c.market, key)) for key in price_fields})
    cost = {f.name: getattr(c.rewards, f.name) / scale for f in fields(c.rewards)
            if f.name not in {"external_cost_per_period", "adjustment_epsilon_kw"}}
    rewards = replace(c.rewards, **cost, adjustment_epsilon_kw=c.rewards.adjustment_epsilon_kw * scale)
    profiles = Profiles(p.loads_kw * scale, p.der_kw * scale, p.grid_cost_per_kwh / scale,
                        p.flex_kw * scale, rigid_kw=p.rigid_kw * scale,
                        start_datetime=system.start_datetime, resolution_minutes=system.resolution_minutes)
    scaled = EnergyMarketEngine(replace(c, system=system, market=market, rewards=rewards), profiles)
    records = []
    for engine, factor in ((original, 1.0), (scaled, scale)):
        initial = (ConsumerState("consumer_0"), ConsumerState("consumer_1"),
                   ConsumerState("consumer_2", tasks=(ShiftTask(-1, 3, .25 * factor),)))
        engine.reset(options={"consumer_states": initial})
        uc = [v * factor for v in (6, 1, 1, 1, 0, 2)]
        actions = {i: [cut * factor, shift * factor, rho, bid / factor]
                   for i, (cut, shift, rho, bid) in zip(engine.consumer_ids,
                       [(1, 1, .5, .7), (1, 0, .5, .65), (0, -1, .4, .6)])}
        records.append(engine.run_period(uc, actions))
    a, b = records
    assert a.rewards == pytest.approx(b.rewards)
    assert a.ledger.cash_profit == pytest.approx(b.ledger.cash_profit)
    assert b.physical.ess_energy_after_kwh == pytest.approx(a.physical.ess_energy_after_kwh * scale)
    assert b.ledger.supply_shortfall_kw == pytest.approx(a.ledger.supply_shortfall_kw * scale)


@pytest.mark.parametrize("price", [.65, 1.7, 3.4, 3.5])
def test_u01_fixed_request_service_threshold(make_engine, price):
    results = []
    for supply in (4, 0):
        engine = make_engine([4], grid_cost=price)
        engine.reset()
        results.append(engine.run_period(U(grid_import_offer_kw=supply, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), {"consumer_0": [0, 0, 0, .3]}))
    full, lost = results
    assert full.rewards["uc"] - lost.rewards["uc"] == pytest.approx(3.4 - price)


def test_a04_physical_shortage_and_voluntary_withholding_are_distinguishable(make_engine):
    maximums, results = [], []
    for grid in (0, 4):
        engine = make_engine([4], grid=grid, ess=0)
        engine.reset()
        maximums.append(float(engine.profiles.der_kw[0]) + grid + min(
            engine.config.system.ess_discharge_max_kw,
            engine.state.ess_energy_kwh * engine.config.system.ess_discharge_efficiency
            / engine.config.system.dt_hours))
        results.append(engine.run_period(U(grid_import_offer_kw=0, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0), {"consumer_0": [0, 0, 0, .3]}))
    assert maximums == [0, 4]
    assert [r.ledger.supply_shortfall_kw for r in results] == [4, 4]
    assert [max(0, r.ledger.supply_reference_kw - p) for r, p in zip(results, maximums)] == [4, 0]
    assert all(r.ledger.supply_shortfall_cost == 2 for r in results)


def test_every_current_observation_coordinate_against_hand_vector(make_engine):
    engine = make_engine([[4, 6]] * 4, der=7, flex=[2, 3], rigid=[2, 3], deadline=2)
    states = (ConsumerState("consumer_0", tasks=(ShiftTask(-2, 0, .25), ShiftTask(-1, 1, .5))),
              ConsumerState("consumer_1", monthly_uc_energy_kwh=120, tasks=(ShiftTask(-2, 0, .5),)))
    uc = engine.reset(options={"consumer_states": states})
    expected_uc = [0, 4, 5, 7, 100, .5, 1, .5, 10, 5, 5, 3, 6 / 13,
                   .75, .5, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_allclose(uc, expected_uc, rtol=1e-7, atol=1e-7)
    observations = engine.apply_uc_action(U(grid_import_offer_kw=8, der_planned_curtailment_kw=1, der_to_ess_kw=1, grid_to_ess_kw=1, ess_discharge_offer_kw=0, dr_procurement_cap_kw=2))
    expected_first = [0, 4, 4, 2, 2, 0, .4, .3, .4, .5, 2, 1, 7, 100, 19,
                      8, 1, 1, 1, 0, 5, 7, 25, .25, .5, 0]
    expected_second = [0, 4, 6, 3, 3, 120, .5, .3, .5, .5, 2, 1, 7, 100, 19,
                       8, 1, 1, 1, 0, 5, 7, 25, .5, 0, 0]
    np.testing.assert_allclose(observations["consumer_0"], expected_first, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(observations["consumer_1"], expected_second, rtol=1e-7, atol=1e-7)
