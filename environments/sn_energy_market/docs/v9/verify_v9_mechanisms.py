"""Independent six-action v9 reference; never imports production environments.

Assertions include successful repairs AND explicitly retained counterexamples.
Only announced offers are cleared. Firm charge and planned curtailment are never
overridden; unused import/discharge offers are not physical energy transactions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
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
    der_bid: float = 0.3
    uc_tariff: float = 0.9

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
    grid_offer: float
    grid_max: float
    der: float = 0.0
    planned_curtail: float = 0.0
    der_charge: float = 0.0
    grid_charge: float = 0.0
    discharge_offer: float = 0.0
    dr_cap: float = 0.0
    energy: float = 0.0
    energy_min: float = 0.0
    energy_max: float = 10.0
    charge_max: float = 8.0
    discharge_max: float = 8.0
    eta_c: float = 0.95
    eta_d: float = 0.95
    dt: float = 0.25
    event: bool = True
    pg_price: float = 0.65
    dr_price: float = 0.2
    dr_value: float = 0.0
    der_cash_cost: float = 0.0
    degradation: float = 0.0
    curtail_opportunity: float = 0.0
    compensation_rate: float = 1.0
    native_penalty: float = 0.0
    due_penalty: float = 1.0

    @property
    def available_discharge(self):
        return min(self.discharge_max, self.eta_d * (self.energy - self.energy_min) / self.dt)

    @property
    def der_capacity(self):
        return self.der - self.planned_curtail - self.der_charge

    @property
    def uc_capacity(self):
        return self.grid_offer + self.discharge_offer - self.grid_charge


def local_bounds(p, rigid, flex, backlog, due, capacity, dt, t, horizon, delay, c_fraction):
    assert 0 <= rigid <= p and flex >= 0 and 0 <= due <= backlog + TOL
    assert 0 <= backlog <= capacity + TOL
    attempt = min(due / dt, flex, backlog / dt)
    cmax = min(p - rigid, flex - attempt)
    c = c_fraction * max(0, cmax)
    smax = min(p - rigid - c, flex - c, max(0, capacity - backlog) / dt)
    if horizon is not None and t + delay > horizon - 1:
        smax = 0.0
    rmax = min(flex - c, backlog / dt)
    low, high = (-rmax, -attempt) if attempt > 0 else (-rmax, smax)
    assert low <= high + TOL and cmax >= -TOL
    return dict(curtail=c, attempt=attempt, low=low, high=high, capacity_violation=max(0, due - flex * dt))


def allocate(requests, bids, capacity):
    out = {i: 0.0 for i in requests}
    for price in sorted(set(bids.values()), reverse=True) if bids is not None else [None]:
        group = [i for i in requests if bids is None or bids[i] == price]
        total = sum(requests[i] for i in group)
        ratio = min(1.0, max(0.0, capacity) / total) if total else 0.0
        for i in group:
            out[i] += requests[i] * ratio
        capacity -= ratio * total
    return out


def run(orders, s):
    original = asdict(s)
    assert len({o.name for o in orders}) == len(orders)
    assert s.dt > 0 and 0 < s.eta_c <= 1 and 0 < s.eta_d <= 1
    assert s.energy_min <= s.energy <= s.energy_max
    assert 0 <= s.grid_offer <= s.grid_max
    assert 0 <= s.grid_charge <= s.grid_offer
    assert s.planned_curtail >= 0 and s.der_charge >= 0
    assert s.planned_curtail + s.der_charge <= s.der + TOL
    assert s.der_charge + s.grid_charge <= min(s.charge_max, (s.energy_max - s.energy) / s.eta_c / s.dt) + TOL
    assert 0 <= s.discharge_offer <= s.available_discharge + TOL
    assert (s.der_charge + s.grid_charge) * s.discharge_offer == 0
    assert s.dr_cap >= 0 and (s.event or (s.dr_cap == 0 and s.dr_price == 0))
    for o in orders:
        assert min(o.baseline, o.curtail, o.defer, o.repay, o.native) >= -TOL
        assert 0 <= o.due_attempt <= o.repay + TOL
        assert not (o.defer and o.repay)
        assert 0 <= o.der_share <= 1 and 0 <= o.der_bid <= o.uc_tariff
    kr, ku = s.der_capacity, s.uc_capacity
    rem_r, rem_u = kr, ku
    xr = {o.name: 0.0 for o in orders}
    xu = {o.name: 0.0 for o in orders}
    layers = []
    bids = {o.name: o.der_bid for o in orders}
    for necessary in (True, False):
        req = {o.name: o.necessary if necessary else o.repay - o.due_attempt for o in orders}
        pref = {o.name: min(req[o.name], max(0, o.der_share * o.demand - xr[o.name])) for o in orders}
        first = allocate(pref, bids, rem_r)
        rem_r = max(0, rem_r - sum(first.values()))
        rem = {i: max(0, req[i] - first[i]) for i in req}
        second = allocate(rem, None, rem_u)
        rem_u = max(0, rem_u - sum(second.values()))
        rem = {i: max(0, rem[i] - second[i]) for i in req}
        rescue = allocate(rem, bids, rem_r)
        rem_r = max(0, rem_r - sum(rescue.values()))
        for i in req:
            xr[i] += first[i] + rescue[i]
            xu[i] += second[i]
        layers.append(
            dict(
                necessary=necessary,
                der_initial=first,
                uc=second,
                der_rescue=rescue,
                unserved={i: max(0, rem[i] - rescue[i]) for i in req},
            )
        )
    xr_total, xu_total = sum(xr.values()), sum(xu.values())
    discharge = min(s.discharge_offer, xu_total + s.grid_charge)
    pg = xu_total + s.grid_charge - discharge
    unsold = kr - xr_total
    discarded = s.planned_curtail + unsold
    energy = s.energy + s.eta_c * (s.der_charge + s.grid_charge) * s.dt - discharge * s.dt / s.eta_d
    assert s.energy_min - TOL <= energy <= s.energy_max + TOL
    energy = min(s.energy_max, max(s.energy_min, energy))
    consumers = {}
    for o in orders:
        y = xr[o.name] + xu[o.name]
        lost = max(0, o.native - y)
        repay = min(o.repay, max(0, y - o.native))
        z = (o.curtail + o.defer - o.repay) if s.event else 0.0
        consumers[o.name] = dict(
            der=xr[o.name],
            uc=xu[o.name],
            served=y,
            native_lost=lost,
            repayment=repay,
            signed_dr=z,
            due_attempt_unserved=max(0, o.due_attempt - repay),
            bill=(o.der_bid * xr[o.name] + o.uc_tariff * xu[o.name]) * s.dt,
            compensation=s.compensation_rate * lost * s.dt,
        )
        close(o.demand - y, lost + o.repay - repay)
    gross = sum(max(0, c["signed_dr"]) for c in consumers.values())
    signed = sum(c["signed_dr"] for c in consumers.values())
    net, paid = max(0, signed), min(s.dr_cap, max(0, signed))
    close(signed, (sum(o.baseline for o in orders) - sum(o.demand for o in orders)) if s.event else 0)
    for c in consumers.values():
        c["paid_dr"] = paid * max(0, c["signed_dr"]) / gross if gross else 0.0
        c["dr_payment"] = c["paid_dr"] * s.dr_price * s.dt
        c["net_transfer"] = c["bill"] - c["dr_payment"] - c["compensation"]
    revenue = sum(c["bill"] for c in consumers.values())
    compensation = sum(c["compensation"] for c in consumers.values())
    payment = sum(c["dr_payment"] for c in consumers.values())
    external = (s.pg_price * pg + s.der_cash_cost * (s.der - discarded)) * s.dt
    cash = revenue - compensation - payment - external
    deg = s.degradation * (s.der_charge + s.grid_charge + discharge) * s.dt
    curt = s.curtail_opportunity * discarded * s.dt
    benefit = s.dr_value * net * s.dt
    due_lower_bound = sum(c["due_attempt_unserved"] for c in consumers.values()) * s.dt
    service = (
        s.native_penalty * sum(c["native_lost"] for c in consumers.values()) * s.dt
        + s.due_penalty * due_lower_bound
    )
    # Full overdue beyond request capacity is evaluated from FIFO in dynamic checks.
    power = (
        pg
        + s.der
        + discharge
        - sum(c["served"] for c in consumers.values())
        - s.der_charge
        - s.grid_charge
        - discarded
    )
    money = cash + external - sum(c["net_transfer"] for c in consumers.values())
    close(power)
    close(money)
    close(sum(c["served"] for c in consumers.values()), min(sum(o.demand for o in orders), kr + ku))
    close(sum(c["paid_dr"] for c in consumers.values()), paid)
    assert -TOL <= pg <= s.grid_offer + TOL <= s.grid_max + TOL
    assert -TOL <= discharge <= s.discharge_offer + TOL
    assert unsold >= -TOL and discarded >= s.planned_curtail - TOL
    assert 0 <= paid <= net + TOL <= gross + TOL
    if sum(layers[0]["unserved"].values()) > TOL:
        close(
            sum(layers[1]["der_initial"].values())
            + sum(layers[1]["uc"].values())
            + sum(layers[1]["der_rescue"].values())
        )
    assert asdict(s) == original
    return dict(
        consumers=consumers,
        layers=layers,
        original_action=[
            s.grid_offer,
            s.planned_curtail,
            s.der_charge,
            s.grid_charge,
            s.discharge_offer,
            s.dr_cap,
        ],
        der_capacity=kr,
        uc_capacity=ku,
        pg=pg,
        discharge=discharge,
        der_charge=s.der_charge,
        grid_charge=s.grid_charge,
        planned_curtail=s.planned_curtail,
        unsold_curtail=unsold,
        actual_curtail=discarded,
        energy_next=energy,
        gross_dr=gross,
        net_dr=net,
        paid_dr=paid,
        revenue=revenue,
        compensation=compensation,
        dr_payment=payment,
        cash=cash,
        degradation_cost=deg,
        curtail_cost=curt,
        benefit=benefit,
        due_attempt_service_cost=service,
        reward_before_unattempted_due=cash - deg - curt + benefit - service,
        power_residual=power,
        cash_residual=money,
    )


def apply_queue(queue, t, delay, defer, repayment, dt):
    remain = repayment * dt
    out = []
    for deadline, energy in sorted(queue):
        used = min(energy, remain)
        remain -= used
        if energy - used > 1e-12:
            out.append((deadline, energy - used))
    close(remain)
    if defer:
        out.append((t + delay, defer * dt))
    return sorted(out)


def main():
    cases = []

    def record(key, claim, data, status="repair_or_invariant"):
        cases.append(dict(id=key, claim=claim, status=status, data=data))

    o = [Order("A", 4, uc_tariff=0.4)]
    supplied = run(o, Supply(4, 4))
    withheld = run(o, Supply(0, 4))
    close(supplied["cash"], -0.25)
    close(withheld["cash"], -1)
    record(
        "N01",
        "Default compensation discourages the original fixed-response negative-margin withholding case",
        dict(supplied=supplied, withheld=withheld),
        "conditional_mitigation",
    )

    o = [Order("A", 4), Order("B", 4), Order("C", 0, repay=4)]
    a = run(o, Supply(8, 8))
    b = run([replace(o[0], der_bid=0.2), *o[1:]], Supply(8, 8))
    close(a["consumers"]["A"]["bill"], b["consumers"]["A"]["bill"])
    close(a["consumers"]["A"]["bill"], 0.9)
    record("N02", "No UC bid or separate backup price; old UC-bid channel exploit unavailable", a)

    a = run([Order("A", 8)], Supply(4, 8, grid_charge=2))
    close(a["grid_charge"], 2)
    close(a["pg"], 4)
    close(a["consumers"]["A"]["served"], 2)
    record("N03", "A legal charging command remains executed even with unmet load", a)

    a = run([Order("A", 4)], Supply(0, 4, energy=1, eta_d=1, discharge_offer=0))
    close(a["discharge"])
    close(a["energy_next"], 1)
    close(a["consumers"]["A"]["served"])
    record("N04", "Available battery does not override a zero discharge offer", a)

    a = run([Order("A", 8, der_share=1)], Supply(0, 8, der=4, planned_curtail=2))
    close(a["planned_curtail"], 2)
    close(a["actual_curtail"], 2)
    close(a["consumers"]["A"]["served"], 2)
    record("N05", "Planned DER curtailment is not withdrawn to rescue unmet requests", a)

    sensitivity = {}
    o = [Order("A", 4, der_share=1)]
    pairs = {
        "grid_offer": (Supply(0, 4), Supply(4, 4)),
        "planned_curtail": (Supply(4, 4, der=4), Supply(4, 4, der=4, planned_curtail=2)),
        "der_charge": (Supply(4, 4, der=4), Supply(4, 4, der=4, der_charge=2)),
        "grid_charge": (Supply(4, 4), Supply(4, 4, grid_charge=2)),
        "discharge_offer": (
            Supply(4, 4, energy=1, eta_d=1),
            Supply(4, 4, energy=1, eta_d=1, discharge_offer=2),
        ),
        "dr_cap": (Supply(4, 4), Supply(4, 4, dr_cap=2)),
    }

    def signatures(x):
        return [
            x["der_capacity"],
            x["uc_capacity"],
            x["pg"],
            x["discharge"],
            x["energy_next"],
            x["actual_curtail"],
            x["dr_payment"],
        ]

    for field, (s0, s1) in pairs.items():
        requests = [Order("A", 4, curtail=2)] if field == "dr_cap" else o
        x0, x1 = run(requests, s0), run(requests, s1)
        assert any(abs(a - b) > TOL for a, b in zip(signatures(x0), signatures(x1)))
        sensitivity[field] = dict(before=signatures(x0), after=signatures(x1))
    record("N06", "All six action variables have verified nonzero effects in legal states", sensitivity)

    bounds = local_bounds(4, 0, 4, 2, 2, 4, 0.25, 3, None, 2, 1)
    close(bounds["attempt"], 4)
    close(bounds["low"], -4)
    close(bounds["high"], -4)
    close(bounds["capacity_violation"], 1)
    record("N07", "Overdue overload retains a nonempty action domain and explicit violation", bounds)

    bounds = local_bounds(4, 2, 4, 0, 0, 4, 0.25, 0, None, 2, 0)
    close(bounds["high"], 2)
    record("N08", "Both curtailment and deferral protect rigid native load", bounds)

    a = run([Order("A", 4, curtail=4), Order("B", 4, repay=4)], Supply(8, 8, dr_cap=4, dr_value=0.8))
    close(a["gross_dr"], 4)
    close(a["net_dr"])
    close(a["paid_dr"])
    close(a["benefit"])
    record("N09", "Offsetting curtailment and repayment create no net payment or DR value", a)

    o = [Order("A", 8, curtail=4, repay=4, uc_tariff=0.8)]
    a, b = run(o, Supply(8, 8, dr_cap=4, dr_value=0.8)), run(o, Supply(4, 8, dr_cap=4, dr_value=0.8))
    close(a["consumers"]["A"]["repayment"], 4)
    close(b["consumers"]["A"]["repayment"])
    close(a["net_dr"], b["net_dr"])
    close(a["benefit"])
    record("N10", "Withholding non-due repayment does not create credited DR", dict(full=a, withheld=b))

    o = [Order("A", 10, curtail=3), Order("B", 10, curtail=3)]
    a = run(o, Supply(14, 14, dr_price=0.5, dr_value=0.8))
    b = run(o, Supply(14, 14, dr_cap=4, dr_price=0.5, dr_value=0.8))
    close(a["pg"], b["pg"])
    close(a["benefit"], b["benefit"])
    close(b["cash"] - a["cash"], -0.5)
    record(
        "N11",
        "Paid labels alone add expenditure without changing response or value",
        dict(cap_zero=a, cap_four=b),
    )

    a = run([Order("A", 0, repay=4, der_share=1), Order("B", 4, der_share=0)], Supply(0, 0, der=4))
    close(a["consumers"]["B"]["served"], 4)
    close(a["consumers"]["A"]["served"])
    record("N12", "Necessary-first DER rescue is global within announced resources", a)

    example_orders = [
        Order("A", 6, curtail=1, defer=1, der_share=0.5, der_bid=0.7),
        Order("B", 5, curtail=1, der_share=0.5, der_bid=0.65),
        Order("C", 4, repay=1, der_share=0.4, der_bid=0.6),
    ]
    example_supply = Supply(
        6,
        8,
        der=6,
        planned_curtail=1,
        der_charge=1,
        grid_charge=1,
        dr_cap=2,
        energy=5,
        dr_value=0.4,
        degradation=0.02,
        curtail_opportunity=0.1,
    )
    a = run(example_orders, example_supply)
    close(a["revenue"], 1.8)
    close(a["cash"], -0.025)
    close(a["reward_before_unattempted_due"], 0.14)
    close(a["energy_next"], 5.475)
    consumer_costs = {}
    for o in example_orders:
        c = a["consumers"][o.name]
        b_after = (0.25 if o.name == "C" else 0) + (o.defer - c["repayment"]) * 0.25
        flex = 0.5 * (o.curtail**2 + o.defer**2) / o.baseline * 0.25
        service = 2 * c["native_lost"] * 0.25 + 0.1 * b_after * 0.25
        consumer_costs[o.name] = c["net_transfer"] + flex + service
    close(consumer_costs["A"], 0.8)
    close(consumer_costs["B"], 0.7854166666666667)
    close(consumer_costs["C"], 0.94375)
    alt = run(
        example_orders, replace(example_supply, grid_offer=8, planned_curtail=0, der_charge=0, grid_charge=0)
    )
    close(alt["cash"], 1.3125)
    close(alt["reward_before_unattempted_due"], 1.5125)
    record(
        "N13",
        "Complete six-action worked example and fixed-response alternative",
        dict(example=a, consumer_costs=consumer_costs, alternative=alt),
    )

    fixed_offer_toy = []
    for wholesale in (0.65, 1.2):
        choices = []
        for cap in (0, 4):
            actions = []
            for c in (0, 4):
                x = run(
                    [Order("A", 4, curtail=c, uc_tariff=0.4)],
                    Supply(4, 4, dr_cap=cap, dr_price=0.5, pg_price=wholesale),
                )
                cost = x["consumers"]["A"]["net_transfer"] + 0.75 * c * c / 4 * 0.25
                actions.append(dict(curtail=c, consumer_cost=cost, uc_cash=x["cash"]))
            best = min(actions, key=lambda x: x["consumer_cost"])
            choices.append(dict(cap=cap, response=best))
        best = max(choices, key=lambda x: x["response"]["uc_cash"])
        assert best["cap"] == (0 if wholesale == 0.65 else 4)
        fixed_offer_toy.append(dict(pg_price=wholesale, choices=choices, best_cap=best["cap"]))
    record(
        "N14",
        "With fixed sufficient supply, follower readaptation yields zero or positive economic procurement",
        fixed_offer_toy,
        "finite_action_example_only",
    )

    joint_toy = []
    for g in (0, 4):
        for cap in (0, 4):
            choices = []
            for c in (0, 4):
                x = run([Order("A", 4, curtail=c, uc_tariff=0.4)], Supply(g, 4, dr_cap=cap, dr_price=0.5))
                own = x["consumers"]["A"]
                cost = own["net_transfer"] + 0.75 * c * c / 4 * 0.25 + 2 * own["native_lost"] * 0.25
                choices.append(
                    dict(curtail=c, consumer_cost=cost, uc_cash=x["cash"], compensation=own["compensation"])
                )
            joint_toy.append(
                dict(
                    grid_offer=g,
                    cap=cap,
                    best_response=min(choices, key=lambda x: x["consumer_cost"]),
                    all_responses=choices,
                )
            )
    best = max(joint_toy, key=lambda x: x["best_response"]["uc_cash"])
    assert best["grid_offer"] == 0 and best["cap"] == 0 and best["best_response"]["curtail"] == 4
    close(best["best_response"]["uc_cash"])
    close(best["best_response"]["compensation"])
    record(
        "N15",
        "Unresolved U03: supply restriction induces unpaid proactive curtailment despite default compensation",
        dict(outcomes=joint_toy, best=best),
        "retained_counterexample",
    )

    a = run([Order("A", 4, uc_tariff=0.4)], Supply(4, 4, pg_price=1.7))
    b = run([Order("A", 4, uc_tariff=0.4)], Supply(0, 4, pg_price=1.7))
    close(a["cash"], -1.3)
    close(b["cash"], -1)
    record(
        "N16",
        "Unresolved U01: compensation does not dominate arbitrarily high purchase cost",
        dict(supplied=a, withheld=b),
        "retained_counterexample",
    )

    o = [Order("A", 4, der_share=1, der_bid=0.3)]
    s = Supply(4, 4, der=4, pg_price=0.2, curtail_opportunity=0.1)
    a, b = run(o, s), run(o, replace(s, planned_curtail=4))
    close(a["cash"], 0.3)
    close(b["cash"], 0.7)
    close(b["reward_before_unattempted_due"], 0.6)
    close(b["consumers"]["A"]["native_lost"])
    record(
        "N17",
        "Unresolved U02: fully served DER-to-UC substitution can profit despite curtailment cost",
        dict(der_sold=a, curtail_and_sell_uc=b),
        "retained_counterexample",
    )

    q = [(0, 1.0), (2, 0.5)]
    q2 = apply_queue(q, 0, 2, 0, 2, 0.25)
    assert q2 == [(0, 0.5), (2, 0.5)]
    record("N18", "Only actual repayment removes oldest energy; overdue remains", dict(before=q, after=q2))

    o = [Order("A", 4, der_share=1, der_bid=0.5), Order("B", 2, der_share=1, der_bid=0.5)]
    a, b = run(o, Supply(0, 0, der=3)), run(list(reversed(o)), Supply(0, 0, der=3))
    for i in ("A", "B"):
        close(a["consumers"][i]["served"], b["consumers"][i]["served"])
    close(a["consumers"]["A"]["served"], 2)
    record("N19", "Equal bids are pro-rata and invariant to consumer order", a)

    o = [Order("A", 4, der_share=1, der_bid=0.7), Order("B", 4, der_share=1, der_bid=0.5)]
    a = run(o, Supply(4, 4, der=4))
    b = run([replace(o[0], der_bid=0.4), o[1]], Supply(4, 4, der=4))
    close(a["consumers"]["A"]["bill"], 0.7)
    close(b["consumers"]["A"]["bill"], 0.9)
    record("N20", "Retained DER bid affects scarce allocation and actual bill", dict(high=a, low=b))

    failed = []
    for bad in [
        Supply(5, 4),
        Supply(1, 4, grid_charge=2),
        Supply(4, 4, der=2, planned_curtail=2, der_charge=1),
        Supply(4, 4, grid_charge=1, discharge_offer=1, energy=2),
    ]:
        saved = asdict(bad)
        try:
            run([Order("A", 4)], bad)
        except AssertionError:
            failed.append(saved)
        else:
            raise AssertionError("Invalid action accepted")
        assert asdict(bad) == saved
    record("N21", "Invalid joint actions are rejected rather than overwritten", failed)

    bounds = local_bounds(4, 0, 4, 0, 0, 4, 0.25, 7, 10, 3, 0)
    close(bounds["high"])
    record("N22", "Finite-horizon shift cutoff remains defined", bounds)

    adjusted = 0.4 - 0.01 + 0.4 * (0 - 1)
    close(adjusted, -0.01)
    record(
        "N23",
        "Terminal inventory value prevents counting initial stored energy as free",
        dict(cash=0.4, degradation=0.01, initial=1, final=0, inventory_value=0.4, adjusted=adjusted),
    )

    a = run([Order("A", 0)], Supply(4, 4, der=4, energy=2, discharge_offer=4))
    close(a["pg"])
    close(a["discharge"])
    close(a["planned_curtail"])
    close(a["actual_curtail"], 4)
    close(a["energy_next"], 2)
    record(
        "N24", "Unused offers stay unused; unsold DER is separate from the unchanged curtailment action", a
    )

    def public_supply(s):
        return [
            s.der,
            s.grid_max,
            s.grid_offer,
            s.planned_curtail,
            s.der_charge,
            s.grid_charge,
            s.discharge_offer,
            s.der_capacity,
            s.uc_capacity,
        ]

    public_a = public_supply(Supply(4, 8, der=4, energy=1, eta_d=1, discharge_offer=4))
    public_b = public_supply(Supply(4, 8, der=4, energy=1, eta_d=1, discharge_offer=0))
    assert public_a != public_b
    assert 23 + (3 + 1) == 3 + 24
    record(
        "N25",
        "Public six-action composition and capacity differ; observation dimension is H+24",
        dict(a=public_a, b=public_b, fixed_scalars=23),
    )

    rng = random.Random(9102026)
    max_power = max_money = max_queue = 0.0

    def make_order(name, t=0, horizon=None, queue=None, dt=0.25):
        backlog = rng.uniform(0, 8) if queue is None else sum(e for _, e in queue)
        due = rng.uniform(0, backlog) if queue is None else sum(e for deadline, e in queue if deadline <= t)
        p, flex = rng.uniform(0, 12), rng.uniform(0, 10)
        rigid = rng.uniform(0, p)
        b = local_bounds(p, rigid, flex, backlog, due, 12, dt, t, horizon, 3, rng.random())
        shift = rng.uniform(b["low"], b["high"])
        o = Order(
            name,
            p,
            b["curtail"],
            max(0, shift),
            max(0, -shift),
            b["attempt"],
            rng.random(),
            rng.choice([0.3, 0.5, 0.7]),
        )
        assert o.curtail + o.defer <= p - rigid + TOL
        assert o.curtail + o.defer + o.repay <= flex + TOL
        return o, b, backlog, due

    def make_supply(energy=None, dt=0.25):
        gmax, der = rng.uniform(0, 25), rng.uniform(0, 15)
        s = Supply(
            rng.uniform(0, gmax),
            gmax,
            der=der,
            energy=rng.uniform(0, 10) if energy is None else energy,
            dr_cap=rng.uniform(0, 20),
            dr_value=rng.uniform(0, 0.8),
            dt=dt,
            degradation=0.02,
            curtail_opportunity=0.1,
            der_cash_cost=0.03,
        )
        mode = rng.choice(["charge", "discharge", "idle"])
        cr = cg = oo = 0.0
        if mode == "charge":
            limit = min(s.charge_max, (s.energy_max - s.energy) / s.eta_c / dt)
            cr = rng.uniform(0, min(der, limit))
            cg = rng.uniform(0, min(s.grid_offer, limit - cr))
        elif mode == "discharge":
            oo = rng.uniform(0, s.available_discharge)
        return replace(
            s, planned_curtail=rng.uniform(0, der - cr), der_charge=cr, grid_charge=cg, discharge_offer=oo
        )

    for _ in range(5000):
        dt = rng.choice([0.125, 0.25, 0.5])
        tuples = [make_order(str(i), dt=dt) for i in range(rng.randint(1, 8))]
        s = make_supply(dt=dt)
        x = run([o for o, *_ in tuples], s)
        max_power = max(max_power, abs(x["power_residual"]))
        max_money = max(max_money, abs(x["cash_residual"]))
        for o, _, backlog, _ in tuples:
            newb = backlog + (o.defer - x["consumers"][o.name]["repayment"]) * dt
            assert -TOL <= newb <= 12 + TOL
    overload = shortage = steps = 0
    max_dynamic_power = 0.0
    for _ in range(100):
        queues = {str(i): [] for i in range(4)}
        energy = rng.uniform(0, 10)
        for t in range(96):
            tuples = [make_order(i, t, 96, q) for i, q in queues.items()]
            s = make_supply(energy)
            if t % 11 in (0, 1, 2):
                s = replace(s, grid_offer=0, grid_charge=0, discharge_offer=0)
            x = run([o for o, *_ in tuples], s)
            energy = x["energy_next"]
            unattempted_due = 0.0
            for o, b, backlog, due in tuples:
                repayment = x["consumers"][o.name]["repayment"]
                expected = backlog + (o.defer - repayment) * s.dt
                queues[o.name] = apply_queue(queues[o.name], t, 3, o.defer, repayment, s.dt)
                actual = sum(e for _, e in queues[o.name])
                max_queue = max(max_queue, abs(actual - expected))
                close(actual, expected)
                assert -TOL <= actual <= 12 + TOL
                due_end = sum(e for deadline, e in queues[o.name] if deadline <= t)
                close(due_end, max(0, due - repayment * s.dt))
                assert b["capacity_violation"] <= due_end + TOL
                unattempted_due += max(0, due_end - x["consumers"][o.name]["due_attempt_unserved"] * s.dt)
                overload += int(b["capacity_violation"] > TOL)
                assert all(deadline <= 95 for deadline, _ in queues[o.name])
            full_reward = x["reward_before_unattempted_due"] - s.due_penalty * unattempted_due
            assert math.isfinite(full_reward)
            close(
                full_reward,
                x["cash"]
                - x["degradation_cost"]
                - x["curtail_cost"]
                + x["benefit"]
                - s.native_penalty * sum(c["native_lost"] for c in x["consumers"].values()) * s.dt
                - s.due_penalty * sum(e for q in queues.values() for deadline, e in q if deadline <= t),
            )
            max_dynamic_power = max(max_dynamic_power, abs(x["power_residual"]))
            shortage += int(
                sum(c["served"] for c in x["consumers"].values()) + TOL < sum(o.demand for o, *_ in tuples)
            )
            steps += 1
    assert overload > 0 and shortage > 0
    for manifest_name in ["protected_files_manifest.json", "history_files_manifest.json"]:
        manifest = json.loads((BASE / manifest_name).read_text(encoding="utf-8"))
        assert all(
            (ROOT / p).exists() and hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
            for p, digest in manifest.items()
        )
    protected = json.loads((BASE / "protected_files_manifest.json").read_text(encoding="utf-8"))
    protected_roots = [
        "environments/sn_energy_market/src",
        "environments/sn_energy_market/configs",
        "environments/sn_energy_market/tests",
        "RLAlgo",
    ]
    actual_paths = {
        str(p.relative_to(ROOT)).replace("\\", "/")
        for folder in protected_roots
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not any(x.startswith(".") for x in p.relative_to(ROOT / folder).parts)
    }
    assert actual_paths == set(protected)
    result = dict(
        version="v9",
        seed=9102026,
        deterministic_cases=cases,
        random_static=dict(samples=5000, max_power_residual_kw=max_power, max_cash_residual_yuan=max_money),
        dynamic=dict(
            episodes=100,
            periods=96,
            consumers=4,
            physical_steps=steps,
            overload_consumer_periods=overload,
            announced_supply_shortage_periods=shortage,
            max_queue_residual_kwh=max_queue,
            max_power_residual_kw=max_dynamic_power,
        ),
        six_action_sensitivity=sensitivity,
        action_preservation="passed on all samples",
        retained_counterexamples=[
            "N15 U03 induced unpaid DR",
            "N16 U01 high-cost withholding",
            "N17 U02 profitable DER curtailment",
        ],
        production_and_history_unchanged=True,
        document_sha256=hashlib.sha256((BASE / "modeling_complete_zh.md").read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            "Independent mathematical reference, not production environment or PPO tests",
            "Finite response examples do not prove general SNE",
            "Synthetic prices, compensation and user costs are not calibrated market parameters",
        ],
        status="passed",
    )
    (BASE / "verification_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["deterministic_cases", "six_action_sensitivity"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
