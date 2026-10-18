"""Hand-solvable checks anchor the independent differential-test oracle."""

import pytest
from v9_oracle import evaluate, local_bounds


def person(**changes):
    values = dict(base=4.0, rigid=0.0, flex=4.0, backlog_limit=25.0,
                  tariff=0.4, bid=0.3, cut=0.0, shift=0.0, rho=0.0)
    values.update(changes)
    return values


def test_u03_hand_calculation_and_existing_response_credit():
    rates = dict(adequacy=2.0, compensation=1.0, loss=2.0, discomfort=0.75)
    full = evaluate([person()], (4, 0, 0, 0, 0, 0), der=0, grid_limit=10,
                    energy=0, grid_price=0.65, rates=rates)
    unpaid = evaluate([person(cut=4)], (0, 0, 0, 0, 0, 0), der=0, grid_limit=10,
                      energy=0, grid_price=0.65, rates=rates)
    paid = evaluate([person(cut=4)], (0, 0, 0, 0, 0, 4), der=0, grid_limit=10,
                    energy=0, grid_price=0.65, rates=rates)
    assert full["cash"] == pytest.approx(-0.25)
    assert unpaid["cash"] == 0
    assert unpaid["reward"] == -2
    assert paid["reward"] == pytest.approx(-0.5)
    assert paid["net_dr"] == unpaid["net_dr"] == 4
    assert unpaid["people"][0]["reward"] == -0.75
    assert paid["people"][0]["reward"] == -0.25


def test_a02_due_extra_reservation_is_preserved():
    p = person(flex=6, cut=4, shift=-2, tasks=(("old", -2, 0, 0.5),))
    result = evaluate([p], (2, 0, 0, 0, 0, 2), der=0, grid_limit=10,
                      energy=0, rates=dict(adequacy=2.0))
    assert result["people"][0]["lost"] == 0
    assert result["people"][0]["backlog"] == 0
    assert result["net_dr"] == 2
    assert result["reference"] == 6
    assert result["adequacy_gap"] == 2
    assert result["adequacy_cost"] == 1


def test_u02_source_substitution():
    p = person(tariff=0.9, rho=1)
    direct = evaluate([p], (0, 0, 0, 0, 0, 0), der=4, grid_limit=10,
                      energy=0, grid_price=0.2, rates=dict(spill=0.1, adequacy=2.0))
    replacement = evaluate([p], (4, 4, 0, 0, 0, 0), der=4, grid_limit=10,
                           energy=0, grid_price=0.2, rates=dict(spill=0.1, adequacy=2.0))
    assert direct["reward"] == pytest.approx(0.3)
    assert replacement["reward"] == pytest.approx(0.6)
    assert direct["people"][0]["energy_cash"] == pytest.approx(0.3)
    assert replacement["people"][0]["energy_cash"] == pytest.approx(0.9)


def test_dr_uses_request_even_when_repayment_unserved():
    ps = [person(cut=4), person(base=0, shift=-4, tasks=(("old", -2, 1, 1.0),))]
    result = evaluate(ps, (0, 0, 0, 0, 0, 8), der=0, grid_limit=10, energy=0)
    assert result["gross_dr"] == 4
    assert result["net_dr"] == result["paid_dr"] == 0
    assert result["people"][1]["repay"] == 0
    assert result["people"][1]["backlog"] == 1


@pytest.mark.parametrize("dt", [0.125, 0.25, 0.5, 1.0])
def test_time_factors_and_terminal_energy(dt):
    p = person(base=0, flex=0, tasks=(("old", -2, 0, 1.0),))
    rates = dict(uc_due=3.0, late=2.0, waiting=1.0, terminal=5.0, uc_terminal=7.0,
                 terminal_energy=0.2, throughput=0.3)
    result = evaluate([p], (2, 0, 0, 2, 0, 0), der=0, grid_limit=10, energy=0,
                      dt=dt, rates=rates, terminal=True, initial_energy=0)
    assert result["due_cost"] == 3
    assert result["people"][0]["service_cost"] == 3 * dt
    assert result["people"][0]["terminal"] == -5
    assert result["throughput_cost"] == pytest.approx(0.6 * dt)
    assert result["terminal"] == pytest.approx(0.2 * 0.95 * 2 * dt - 7)


def test_local_due_bounds_do_not_allow_cut_to_remove_due_obligation():
    p = person(tasks=(("old", -2, 0, 2.0),))
    due, cut_upper, shift = local_bounds(p, period=0, deadline=4, horizon=1, dt=0.25)
    assert due == 4
    assert cut_upper == 0
    assert shift(0) == (-4, -4)


def test_price_tie_proportional_and_necessary_layer_before_early():
    ps = [person(base=4, rho=1, bid=0.3), person(base=2, rho=1, bid=0.3),
          person(base=0, rho=1, bid=0.9, shift=-4, tasks=(("early", -1, 3, 1),))]
    result = evaluate(ps, (0, 0, 0, 0, 0, 0), der=3, grid_limit=10, energy=0)
    assert [p["der"] for p in result["people"]] == [2, 1, 0]
    assert result["layers"][1]["der_initial"] == [0, 0, 0]
