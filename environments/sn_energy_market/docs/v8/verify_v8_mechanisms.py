"""Independent v8 formula checks. Never imports or modifies the production env.

This finite reference calculation verifies declared equations and counterexamples;
it is neither an RL environment nor an equilibrium/convergence certification.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).parent
TOL = 1e-8


def close(a, b=0.0):
    assert math.isclose(a, b, abs_tol=TOL, rel_tol=TOL), (a, b)


@dataclass(frozen=True)
class Order:
    name: str
    baseline: float
    curtail: float = 0.0
    defer: float = 0.0
    repay: float = 0.0
    due_attempt: float = 0.0
    der_share: float = 0.0
    der_bid: float = 0.30
    uc_tariff: float = 0.90

    @property
    def native(self):
        return self.baseline - self.curtail - self.defer

    @property
    def demand(self):
        return self.native + self.repay

    @property
    def necessary(self):
        return self.native + self.due_attempt


@dataclass(frozen=True)
class Supply:
    pg_max: float
    der: float = 0.0
    pg_price: float = 0.65
    der_charge_plan: float = 0.0
    pg_charge_plan: float = 0.0
    discharge_plan: float = 0.0
    energy: float = 0.0
    energy_min: float = 0.0
    energy_max: float = 10.0
    charge_max: float = 8.0
    discharge_max: float = 8.0
    eta_c: float = 0.95
    eta_d: float = 0.95
    dt: float = 0.25
    event: bool = True
    dr_cap: float = 0.0
    dr_price: float = 0.20
    dr_value: float = 0.0
    der_cash_cost: float = 0.0
    degradation: float = 0.0
    curtail_opportunity: float = 0.0
    compensation_rate: float = 0.0
    native_penalty: float = 0.0

    @property
    def available_discharge(self):
        return min(self.discharge_max, self.eta_d * (self.energy - self.energy_min) / self.dt)


def local_bounds(baseline, rigid, flex, backlog, due_kwh, backlog_max, dt, t, horizon, delay, c_fraction):
    """Equations 1-5, prior to action sampling; horizon=None means continuing."""
    assert 0 <= rigid <= baseline and flex >= 0
    assert 0 <= due_kwh <= backlog + TOL and 0 <= backlog <= backlog_max + TOL
    attempt = min(due_kwh / dt, flex, backlog / dt)
    c_max = min(baseline - rigid, flex - attempt)
    c = c_fraction * max(0.0, c_max)
    shift_max = min(baseline - rigid - c, flex - c, max(0.0, backlog_max - backlog) / dt)
    if horizon is not None and t + delay > horizon - 1:
        shift_max = 0.0
    repay_max = min(flex - c, backlog / dt)
    low, high = (-repay_max, -attempt) if attempt > 0 else (-repay_max, shift_max)
    assert c_max >= -TOL and low <= high + TOL
    return dict(
        curtail=c,
        c_max=c_max,
        shift_low=low,
        shift_high=high,
        attempt=attempt,
        due_capacity_gap=max(0, due_kwh - flex * dt),
    )


def allocate(requests, bids, capacity):
    """bids=None: pro-rata. Otherwise descending bids, pro-rata exact ties."""
    out = {key: 0.0 for key in requests}
    for price in sorted(set(bids.values()), reverse=True) if bids is not None else [None]:
        group = [key for key in requests if bids is None or bids[key] == price]
        total = sum(requests[key] for key in group)
        ratio = min(1.0, max(0.0, capacity) / total) if total else 0.0
        for key in group:
            out[key] += requests[key] * ratio
        capacity -= total * ratio
    return out


def run(orders, supply):
    """Equations 6-26; queue- and terminal-specific costs are checked separately."""
    s = supply
    assert len({o.name for o in orders}) == len(orders)
    assert s.energy_min <= s.energy <= s.energy_max
    assert s.pg_max >= 0 and s.der >= 0 and s.dt > 0
    assert 0 < s.eta_c <= 1 and 0 < s.eta_d <= 1
    assert 0 <= s.der_charge_plan <= s.der and 0 <= s.pg_charge_plan <= s.pg_max
    cp = s.der_charge_plan + s.pg_charge_plan
    assert cp <= min(s.charge_max, (s.energy_max - s.energy) / s.eta_c / s.dt) + TOL
    assert 0 <= s.discharge_plan <= s.available_discharge + TOL
    assert cp * s.discharge_plan == 0
    assert s.dr_cap >= 0 and (s.event or (s.dr_cap == 0 and s.dr_price == 0))
    for o in orders:
        assert min(o.baseline, o.curtail, o.defer, o.repay, o.native) >= -TOL
        assert 0 <= o.due_attempt <= o.repay + TOL
        assert not (o.defer and o.repay)
        assert 0 <= o.der_share <= 1 and 0 <= o.der_bid <= o.uc_tariff
    demand = sum(o.demand for o in orders)
    offer = max(s.discharge_plan, min(s.available_discharge, max(0.0, demand - s.der - s.pg_max)))
    if offer > 0:
        cr = cg = 0.0
    else:
        budget = min(cp, max(0.0, s.der + s.pg_max - demand))
        cr = min(s.der_charge_plan, budget)
        cg = min(s.pg_charge_plan, max(0.0, budget - cr))
    kr, ku = s.der - cr, s.pg_max - cg + offer
    rem_r, rem_u = kr, ku
    xr = {o.name: 0.0 for o in orders}
    xu = {o.name: 0.0 for o in orders}
    layers = []
    bids = {o.name: o.der_bid for o in orders}
    for necessary in (True, False):
        req = {o.name: o.necessary if necessary else o.repay - o.due_attempt for o in orders}
        preferred = {o.name: min(req[o.name], max(0.0, o.der_share * o.demand - xr[o.name])) for o in orders}
        first = allocate(preferred, bids, rem_r)
        rem_r = max(0.0, rem_r - sum(first.values()))
        after_first = {i: max(0.0, req[i] - first[i]) for i in req}
        second = allocate(after_first, None, rem_u)
        rem_u = max(0.0, rem_u - sum(second.values()))
        after_second = {i: max(0.0, after_first[i] - second[i]) for i in req}
        rescue = allocate(after_second, bids, rem_r)
        rem_r = max(0.0, rem_r - sum(rescue.values()))
        for i in req:
            xr[i] += first[i] + rescue[i]
            xu[i] += second[i]
        layers.append(
            dict(
                necessary=necessary,
                requests=req,
                der_preferred=first,
                uc=second,
                der_rescue=rescue,
                unserved={i: max(0.0, after_second[i] - rescue[i]) for i in req},
            )
        )
    xu_total, xr_total = sum(xu.values()), sum(xr.values())
    discharge = min(offer, xu_total)
    pg = xu_total + cg - discharge
    curtail = s.der - cr - xr_total
    new_energy = s.energy + s.eta_c * (cr + cg) * s.dt - discharge * s.dt / s.eta_d
    # Normalize only a validated floating-point boundary residual before reuse.
    assert s.energy_min - TOL <= new_energy <= s.energy_max + TOL
    new_energy = min(s.energy_max, max(s.energy_min, new_energy))
    consumers = {}
    for o in orders:
        served = xr[o.name] + xu[o.name]
        native_lost = max(0.0, o.native - served)
        repayment = min(o.repay, max(0.0, served - o.native))
        requested_dr = (o.curtail + o.defer - o.repay) if s.event else 0.0
        consumers[o.name] = dict(
            der=xr[o.name],
            uc=xu[o.name],
            served=served,
            native_lost=native_lost,
            repay_served=repayment,
            signed_request_dr=requested_dr,
            bill=(o.der_bid * xr[o.name] + o.uc_tariff * xu[o.name]) * s.dt,
            compensation=s.compensation_rate * native_lost * s.dt,
        )
        close(o.demand - served, native_lost + o.repay - repayment)
    gross = sum(max(0.0, x["signed_request_dr"]) for x in consumers.values())
    signed = sum(x["signed_request_dr"] for x in consumers.values())
    close(signed, (sum(o.baseline for o in orders) - demand) if s.event else 0.0)
    net = max(0.0, signed)
    paid = min(s.dr_cap, net)
    for x in consumers.values():
        x["paid_dr"] = paid * max(0.0, x["signed_request_dr"]) / gross if gross else 0.0
        x["incentive"] = x["paid_dr"] * s.dr_price * s.dt
        x["net_transfer_to_uc"] = x["bill"] - x["incentive"] - x["compensation"]
    revenue = sum(x["bill"] for x in consumers.values())
    dr_payment = sum(x["incentive"] for x in consumers.values())
    compensation = sum(x["compensation"] for x in consumers.values())
    external = (s.pg_price * pg + s.der_cash_cost * (s.der - curtail)) * s.dt
    cash = revenue - dr_payment - compensation - external
    deg_cost = s.degradation * (cr + cg + discharge) * s.dt
    curt_cost = s.curtail_opportunity * curtail * s.dt
    native_loss = sum(x["native_lost"] for x in consumers.values())
    benefit = s.dr_value * net * s.dt
    reward = cash - deg_cost - curt_cost + benefit - s.native_penalty * native_loss * s.dt
    served_total = sum(x["served"] for x in consumers.values())
    expected_service = min(demand, s.der + s.pg_max + s.available_discharge)
    power_residual = pg + s.der + discharge - served_total - cr - cg - curtail
    money_residual = cash + external - sum(x["net_transfer_to_uc"] for x in consumers.values())
    close(power_residual)
    close(money_residual)
    close(served_total, expected_service)
    close(sum(x["paid_dr"] for x in consumers.values()), paid)
    assert -TOL <= pg <= s.pg_max + TOL
    assert curtail >= -TOL and s.energy_min - TOL <= new_energy <= s.energy_max + TOL
    assert -TOL <= paid <= net + TOL and net <= gross + TOL
    assert (cr + cg) * discharge <= TOL
    if sum(layers[0]["unserved"].values()) > TOL:
        close(
            sum(layers[1]["der_preferred"].values())
            + sum(layers[1]["uc"].values())
            + sum(layers[1]["der_rescue"].values())
        )
    return dict(
        consumers=consumers,
        layers=layers,
        pg=pg,
        der_charge=cr,
        pg_charge=cg,
        discharge=discharge,
        discharge_offer=offer,
        der_curtail=curtail,
        energy_next=new_energy,
        gross_dr=gross,
        signed_dr=signed,
        net_dr=net,
        paid_dr=paid,
        revenue=revenue,
        dr_payment=dr_payment,
        cash=cash,
        benefit=benefit,
        reward=reward,
        served=served_total,
        power_residual=power_residual,
        money_residual=money_residual,
    )


def apply_queue(queue, t, delay, defer_kw, served_repay_kw, dt):
    remaining = served_repay_kw * dt
    result = []
    for deadline, energy in sorted(queue):
        used = min(energy, remaining)
        remaining -= used
        if energy - used > 1e-12:
            result.append((deadline, energy - used))
    close(remaining)
    if defer_kw > 0:
        result.append((t + delay, defer_kw * dt))
    return sorted(result)


def make_order(rng, name, t=0, horizon=None, delay=3, queue=None, dt=0.25):
    if queue is None:
        backlog = rng.uniform(0, 8)
        due = rng.uniform(0, backlog)
    else:
        backlog = sum(e for _, e in queue)
        due = sum(e for deadline, e in queue if deadline <= t)
    p, flex = rng.uniform(0, 12), rng.uniform(0, 10)
    rigid = rng.uniform(0, p)
    bounds = local_bounds(p, rigid, flex, backlog, due, 12, dt, t, horizon, delay, rng.random())
    shift = rng.uniform(bounds["shift_low"], bounds["shift_high"])
    o = Order(
        name,
        p,
        bounds["curtail"],
        max(0.0, shift),
        max(0.0, -shift),
        bounds["attempt"],
        rng.random(),
        rng.choice([0.3, 0.5, 0.7]),
        0.9,
    )
    assert o.curtail + o.defer <= p - rigid + TOL
    assert o.curtail + o.defer + o.repay <= flex + TOL
    return o, bounds, backlog, due


def make_supply(rng, energy=None, dt=0.25):
    s = Supply(
        rng.uniform(0, 25),
        der=rng.uniform(0, 15),
        energy=rng.uniform(0, 10) if energy is None else energy,
        dt=dt,
        dr_cap=rng.uniform(0, 20),
        dr_value=rng.uniform(0, 0.8),
        der_cash_cost=0.03,
        degradation=0.01,
        curtail_opportunity=0.02,
        compensation_rate=0.1,
    )
    mode = rng.choice(["charge", "discharge", "idle"])
    if mode == "charge":
        limit = min(s.charge_max, (s.energy_max - s.energy) / s.eta_c / dt)
        cr = rng.uniform(0, min(s.der, limit))
        cg = rng.uniform(0, min(s.pg_max, limit - cr))
        s = replace(s, der_charge_plan=cr, pg_charge_plan=cg)
    if mode == "discharge":
        s = replace(s, discharge_plan=rng.uniform(0, s.available_discharge))
    return s


def main():
    cases = []

    def record(key, claim, data, kind="repair_or_invariant"):
        cases.append(dict(id=key, claim=claim, kind=kind, data=data))

    # Old C01/C10: no UC supply-withholding variable, even when serving loses cash.
    a = run([Order("A", 4, uc_tariff=0.4)], Supply(4, pg_charge_plan=4))
    close(a["cash"], -0.25)
    close(a["served"], 4)
    close(a["pg_charge"])
    record("V01", "Negative margin does not authorize withholding available PG energy", a)

    orders = [Order("A", 4, uc_tariff=0.9), Order("B", 4), Order("C", 0, repay=4)]
    a = run(orders, Supply(8))
    b = run([replace(orders[0], der_bid=0.2), *orders[1:]], Supply(8))
    close(a["consumers"]["A"]["bill"], 0.9)
    close(a["consumers"]["A"]["bill"], b["consumers"]["A"]["bill"])
    close(a["consumers"]["C"]["served"])
    record(
        "V02",
        "Old UC-bid fallback exploit removed: UC price is not a consumer action",
        dict(before=a, after=b),
    )

    trace = []
    for t, due in [(2, 1), (3, 2)]:
        x = local_bounds(4, 0, 4, 2, due, 4, 0.25, t, None, 2, 1)
        close(x["attempt"], 4)
        close(x["shift_low"], -4)
        close(x["shift_high"], -4)
        trace.append(dict(t=t, due=due, bounds=x))
    close(trace[-1]["bounds"]["due_capacity_gap"], 1)
    record("V03", "Reachable overdue overload has a nonempty action domain and explicit violation", trace)

    a = run([Order("A", 4, curtail=4), Order("B", 4, repay=4)], Supply(8, dr_cap=4, dr_value=0.8))
    close(a["gross_dr"], 4)
    close(a["net_dr"])
    close(a["paid_dr"])
    close(a["benefit"])
    record("V04", "Offsetting reduction and repayment create no net DR payment or value", a)

    o = [Order("A", 8, curtail=4, repay=4, uc_tariff=0.8)]
    a, b = (run(o, Supply(g, dr_cap=4, dr_value=0.8)) for g in [8, 4])
    close(a["consumers"]["A"]["repay_served"], 4)
    close(b["consumers"]["A"]["repay_served"])
    close(a["net_dr"], b["net_dr"])
    close(a["benefit"])
    record(
        "V05",
        "Exogenous non-delivery does not create DR; UC cannot choose an artificial PG cap",
        dict(full=a, physical_shortage=b),
    )

    o = [Order("A", 10, curtail=3), Order("B", 10, curtail=3)]
    s = Supply(14, dr_price=0.5, dr_value=0.8)
    a, b = run(o, s), run(o, replace(s, dr_cap=4))
    close(a["pg"], b["pg"])
    close(a["benefit"], b["benefit"])
    close(b["reward"] - a["reward"], -0.5)
    record("V06", "Paid labels alone only add real compensation expenditure", dict(cap_zero=a, cap_four=b))

    a = run([Order("A", 0, repay=4, der_share=1), Order("B", 4, der_share=0)], Supply(0, der=4))
    close(a["consumers"]["A"]["served"])
    close(a["consumers"]["B"]["served"], 4)
    record("V07", "DER rescue serves necessary load before another consumer optional repayment", a)

    o = [
        Order("A", 6, curtail=1, defer=1, der_share=0.5, der_bid=0.7),
        Order("B", 5, curtail=1, der_share=0.5, der_bid=0.65),
        Order("C", 4, repay=1, der_share=0.4, der_bid=0.6),
    ]
    a = run(o, Supply(8, der=4, dr_cap=2, dr_value=0.4))
    close(a["revenue"], 2.475)
    close(a["cash"], 1.075)
    close(a["reward"], 1.275)
    close(a["net_dr"], 2)
    record("V08", "Complete document worked example", a)

    response_comparisons = []
    for wholesale in [0.65, 1.2]:
        outcomes = []
        for cap in [0, 4]:
            choices = []
            for c in [0, 4]:
                x = run(
                    [Order("A", 4, curtail=c, uc_tariff=0.4)],
                    Supply(4, pg_price=wholesale, dr_cap=cap, dr_price=0.5),
                )
                consumer_cost = x["consumers"]["A"]["net_transfer_to_uc"] + 0.75 * c * c / 4 * 0.25
                choices.append(dict(curtail=c, consumer_cost=consumer_cost, uc_cash=x["cash"]))
            best = min(choices, key=lambda x: x["consumer_cost"])
            outcomes.append(dict(cap=cap, consumer_best_response=best, all_consumer_choices=choices))
        best_uc = max(outcomes, key=lambda x: x["consumer_best_response"]["uc_cash"])
        assert best_uc["cap"] == (0 if wholesale == 0.65 else 4)
        response_comparisons.append(
            dict(pg_price=wholesale, outcomes=outcomes, uc_optimum_in_two_action_set=best_uc["cap"])
        )
    record(
        "V09",
        "With follower readaptation, zero and positive procurement both arise without DR utility subsidy",
        response_comparisons,
        "finite_two_action_stackelberg_example_not_general_equilibrium",
    )

    a = run([Order("A", 8)], Supply(4, energy=2, pg_charge_plan=4, eta_d=1, discharge_max=4))
    close(a["served"], 8)
    close(a["pg"], 4)
    close(a["discharge"], 4)
    close(a["pg_charge"])
    record("V10", "Charging yields and physically available ESS fills a real shortage", a)

    close(0.8 * 2**2 / 4, 0.8)
    close(0.8 * (2 / 4) ** 2, 0.2)
    record(
        "V11",
        "Original Eq10 denominator is not squared; v8 linear value is a new assumption",
        dict(original_positive_term=0.8, misquoted_term=0.2),
        "source_formula_arithmetic",
    )

    terminal_adjusted = 0.4 - 0.01 + 0.4 * (0 - 1)
    close(terminal_adjusted, -0.01)
    record(
        "V12",
        "Stored initial energy is not free in finite-horizon inventory-adjusted reward",
        dict(
            cash=0.4,
            degradation=0.01,
            initial_energy=1,
            final_energy=0,
            inventory_value=0.4,
            adjusted_reward=terminal_adjusted,
        ),
    )

    public_a = [4, 4, 0, 0, 4]  # PG max, discharge available, charge R/G, discharge plan
    public_b = [4, 0, 0, 0, 0]
    assert public_a != public_b
    record(
        "V13",
        "Published source composition distinguishes old aliased supply states",
        dict(ess_state=public_a, pg_only_state=public_b),
    )

    x = local_bounds(4, 2, 4, 0, 0, 4, 0.25, 0, None, 2, 0)
    close(x["shift_high"], 2)
    record("V14", "Rigid native load is protected against deferral as well as curtailment", x)

    high_bill, lower_partial_bill = 4 * 0.95 * 0.25, (3 * 0.80 + 1 * 1.00) * 0.25
    close(high_bill - lower_partial_bill, 0.10)
    # Realizable pro-rata tie: A lowers to B's 0.8. UC cap=4.5, A=4, B=2.
    high = allocate({"A": 4, "B": 2}, {"A": 0.95, "B": 0.8}, 4.5)
    low = allocate({"A": 4, "B": 2}, {"A": 0.8, "B": 0.8}, 4.5)
    close(high["A"], 4)
    close(low["A"], 3)
    record(
        "V15",
        "Rejected patch: ceiling-price backup still permits partial-fill price reduction",
        dict(high_normal=high, low_normal=low, high_bill=high_bill, low_bill=lower_partial_bill),
        "counterexample_to_rejected_v7_patch",
    )

    queue = [(0, 1.0), (2, 0.5)]
    new_queue = apply_queue(queue, 0, 2, 0, 2, 0.25)
    assert new_queue == [(0, 0.5), (2, 0.5)]
    close(sum(e for _, e in new_queue), 1)
    record(
        "V16",
        "Only actual repayment removes oldest energy; overdue remains explicit",
        dict(before=queue, after=new_queue),
    )

    o = [Order("A", 4, der_share=1, der_bid=0.5), Order("B", 2, der_share=1, der_bid=0.5)]
    a, b = run(o, Supply(0, der=3)), run(list(reversed(o)), Supply(0, der=3))
    for key in ["A", "B"]:
        close(a["consumers"][key]["served"], b["consumers"][key]["served"])
    close(a["consumers"]["A"]["served"], 2)
    close(a["consumers"]["B"]["served"], 1)
    record("V17", "Equal bids and equal UC priorities use pro-rata, invariant to consumer ordering", a)

    x = local_bounds(4, 0, 4, 0, 0, 4, 0.25, 7, 10, 3, 0)
    close(x["shift_high"])
    record("V18", "Finite-horizon shift cutoff forbids creating tasks due after the last period", x)

    frozen_price, month_before, billed_kwh = 0.6, 99.9, 0.5
    bill = frozen_price * billed_kwh
    month_settled, month_next = month_before + billed_kwh, 0.0
    close(bill, 0.3)
    record(
        "V19",
        "Month-end bills use frozen price, then update and reset next-month cumulative energy",
        dict(
            price=frozen_price, before=month_before, bill=bill, settled=month_settled, next_month=month_next
        ),
        "accounting_boundary",
    )

    x = local_bounds(0, 0, 0, 0, 0, 0, 0.25, 0, None, 1, 0.5)
    close(x["shift_low"], x["shift_high"])
    a = run([Order("A", 0)], Supply(0, event=False, dr_price=0, energy_max=0))
    close(a["reward"])
    record(
        "V20",
        "Zero-width actions, empty demand/resources, and non-event DR are well defined",
        dict(bounds=x, market=a),
    )

    o = [Order("A", 4, der_share=1, der_bid=0.7), Order("B", 4, der_share=1, der_bid=0.5)]
    a = run(o, Supply(4, der=4))
    b = run([replace(o[0], der_bid=0.4), o[1]], Supply(4, der=4))
    close(a["consumers"]["A"]["der"], 4)
    close(b["consumers"]["A"]["der"], 0)
    close(a["consumers"]["A"]["bill"], 0.7)
    close(b["consumers"]["A"]["bill"], 0.9)
    record(
        "V21", "Retained DER bid changes scarce allocation and actual payment", dict(high_bid=a, low_bid=b)
    )

    # Random legal single periods, with independent expected maximal-service formula.
    rng = random.Random(8102026)
    max_power = max_money = max_queue = max_soc = 0.0
    for _ in range(5000):
        dt = rng.choice([0.125, 0.25, 0.5])
        triples = [make_order(rng, str(i), dt=dt) for i in range(rng.randint(1, 8))]
        s = make_supply(rng, dt=dt)
        x = run([r[0] for r in triples], s)
        max_power = max(max_power, abs(x["power_residual"]))
        max_money = max(max_money, abs(x["money_residual"]))
        for o, _, backlog, _ in triples:
            new_b = backlog + (o.defer - x["consumers"][o.name]["repay_served"]) * dt
            assert -TOL <= new_b <= 12 + TOL
        expected_e = (
            s.energy + s.eta_c * (x["der_charge"] + x["pg_charge"]) * dt - x["discharge"] * dt / s.eta_d
        )
        max_soc = max(max_soc, abs(x["energy_next"] - expected_e))

    # Independent dynamic FIFO lists with alternating ample/zero generation.
    dynamic_steps = 0
    due_overload_steps = 0
    shortage_steps = 0
    max_dynamic_power = 0.0
    for episode in range(100):
        queues = {str(i): [] for i in range(4)}
        energy = rng.uniform(0, 10)
        for t in range(96):
            triples = [make_order(rng, name, t=t, horizon=96, delay=3, queue=q) for name, q in queues.items()]
            s = make_supply(rng, energy=energy)
            if t % 11 in (0, 1, 2):
                s = replace(s, pg_max=0, der=0, der_charge_plan=0, pg_charge_plan=0, discharge_plan=0)
            x = run([r[0] for r in triples], s)
            energy = x["energy_next"]
            for o, bounds, backlog, due in triples:
                delivered = x["consumers"][o.name]["repay_served"]
                expected_b = backlog + (o.defer - delivered) * s.dt
                queues[o.name] = apply_queue(queues[o.name], t, 3, o.defer, delivered, s.dt)
                actual_b = sum(e for _, e in queues[o.name])
                max_queue = max(max_queue, abs(actual_b - expected_b))
                close(actual_b, expected_b)
                assert -TOL <= actual_b <= 12 + TOL
                due_end = sum(e for deadline, e in queues[o.name] if deadline <= t)
                close(due_end, max(0.0, due - delivered * s.dt))
                assert bounds["due_capacity_gap"] <= due_end + TOL
                due_overload_steps += int(bounds["due_capacity_gap"] > TOL)
                assert all(deadline <= 95 for deadline, _ in queues[o.name])
            max_dynamic_power = max(max_dynamic_power, abs(x["power_residual"]))
            shortage_steps += int(x["served"] + TOL < sum(o.demand for o, *_ in triples))
            dynamic_steps += 1
    assert due_overload_steps > 0 and shortage_steps > 0

    protected = json.loads((BASE / "protected_files_manifest.json").read_text(encoding="utf-8"))
    protected_changes = [
        name
        for name, digest in protected.items()
        if not (ROOT / name).exists() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    assert not protected_changes, protected_changes
    protected_roots = [
        "environments/sn_energy_market/src",
        "environments/sn_energy_market/configs",
        "environments/sn_energy_market/tests",
        "RLAlgo",
    ]
    actual_protected_names = {
        str(p.relative_to(ROOT)).replace("\\", "/")
        for root in protected_roots
        for p in (ROOT / root).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not any(x.startswith(".") for x in p.relative_to(ROOT / root).parts)
    }
    assert actual_protected_names == set(protected)
    sources = {
        "v7_snapshot": BASE.parent / "v7/source_v7_supplied.md",
        "v7_integrated": BASE.parent / "v7/modeling_integrated_zh.md",
        "v8_document": BASE / "modeling_complete_zh.md",
        "specified_download": Path(
            r"C:\Users\ykq31\Downloads\Stackelberg_Nash_Dual_PAB_DR_Modeling_ZH_v7.md"
        ),
    }
    manifest = {
        key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for key, path in sources.items()
    }
    assert manifest["v7_snapshot"]["sha256"] == manifest["specified_download"]["sha256"]
    result = dict(
        version="v8",
        seed=8102026,
        deterministic_cases=cases,
        random_static=dict(
            samples=5000,
            max_power_residual_kw=max_power,
            max_cash_residual_yuan=max_money,
            max_soc_residual_kwh=max_soc,
        ),
        dynamic=dict(
            episodes=100,
            periods_per_episode=96,
            consumers=4,
            physical_steps=dynamic_steps,
            consumer_periods=dynamic_steps * 4,
            overload_consumer_periods=due_overload_steps,
            physical_shortage_periods=shortage_steps,
            max_queue_residual_kwh=max_queue,
            max_power_residual_kw=max_dynamic_power,
        ),
        protected_production_files=dict(count=len(protected), changed=protected_changes),
        source_manifest=manifest,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            "independent mathematical reference, not production environment tests",
            "no learning, general Nash or Stackelberg equilibrium certification",
            "synthetic scenarios, no empirical tariff or DR baseline calibration",
        ],
        status="passed",
    )
    (BASE / "verification_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["deterministic_cases", "source_manifest"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
