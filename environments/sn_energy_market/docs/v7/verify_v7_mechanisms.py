"""Independent executable checks of the supplied v7 formulas; not a v7 environment.

Only Python's standard library is used. Counterexample assertions PASS when the
reported undesirable outcome is reproduced, not when the mechanism is safe.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

DT = 0.25
ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("verification_results.json")


@dataclass(frozen=True)
class Order:
    name: str
    baseline: float
    curtail: float = 0.0
    defer: float = 0.0
    repay: float = 0.0
    due: float = 0.0
    der_share: float = 0.0
    uc_bid: float = 0.8
    der_bid: float = 0.3

    @property
    def native(self):
        return self.baseline - self.curtail - self.defer

    @property
    def demand(self):
        return self.native + self.repay

    @property
    def mandatory(self):
        return self.native + self.due


@dataclass(frozen=True)
class Supply:
    pg_plan: float
    pg_max: float
    der_generation: float = 0.0
    der_curtail_plan: float = 0.0
    der_charge: float = 0.0
    pg_charge: float = 0.0
    discharge: float = 0.0
    backup_contract: float = 0.0
    pg_price: float = 0.65
    backup_extra: float = 0.30
    dr_cap: float = 0.0
    dr_price: float = 0.20
    dr_reference: float = 4.0
    dr_beta: float = 0.80


def close(x, y=0.0):
    assert math.isclose(x, y, abs_tol=1e-9, rel_tol=1e-9), (x, y)


def allocate(requests, bids, capacity, auction):
    """Two priority classes; exact-price ties pro rata; reserve ignores bids."""
    assert all(q >= -1e-9 for parts in requests.values() for q in parts)
    requests = {i: [max(0.0, q) for q in parts] for i, parts in requests.items()}
    result = {key: [0.0, 0.0] for key in requests}
    for priority in (0, 1):
        prices = sorted(set(bids.values()), reverse=True) if auction else [None]
        for price in prices:
            group = [i for i in requests if price is None or bids[i] == price]
            total = sum(requests[i][priority] for i in group)
            ratio = min(1.0, max(0.0, capacity) / total) if total else 0.0
            for i in group:
                result[i][priority] = requests[i][priority] * ratio
            capacity -= total * ratio
    return result


def run(orders, supply):
    """v7 sections 3-10, excluding unspecified noncash/terminal costs."""
    requests = {}
    for o in orders:
        assert o.native >= 0 and 0 <= o.due <= o.repay
        assert not (o.defer and o.repay)
        der_initial = o.der_share * o.demand
        requests[o.name] = [min(der_initial, o.mandatory), max(0, der_initial - o.mandatory)]
    der_cap = supply.der_generation - supply.der_curtail_plan - supply.der_charge
    uc_cap = supply.pg_plan + supply.discharge - supply.pg_charge
    assert min(der_cap, uc_cap) >= -1e-9
    der = allocate(requests, {o.name: o.der_bid for o in orders}, der_cap, True)
    remaining = {o.name: [o.mandatory - der[o.name][0], o.repay - o.due - der[o.name][1]] for o in orders}
    uc = allocate(remaining, {o.name: o.uc_bid for o in orders}, uc_cap, True)
    ordinary = sum(sum(x) for x in uc.values())
    discharge = min(supply.discharge, ordinary + supply.pg_charge)
    pg_base = ordinary + supply.pg_charge - discharge
    backup_cap = min(supply.backup_contract, supply.pg_max - pg_base)
    assert backup_cap >= -1e-9
    unfilled = {i: [remaining[i][j] - uc[i][j] for j in (0, 1)] for i in remaining}
    backup = allocate(unfilled, {o.name: o.uc_bid for o in orders}, backup_cap, False)
    consumers = {}
    for o in orders:
        r, u, b = sum(der[o.name]), sum(uc[o.name]), sum(backup[o.name])
        served = r + u + b
        native_lost = o.native - min(o.native, served)
        repayment = min(o.repay, max(0.0, served - o.native))
        signed_dr = o.curtail + o.defer - repayment
        verified = max(0.0, signed_dr)
        bill = (o.der_bid * r + o.uc_bid * (u + b)) * DT
        consumers[o.name] = dict(
            der=r,
            uc=u,
            backup=b,
            served=served,
            native_lost=native_lost,
            repay_served=repayment,
            repay_unfilled=o.repay - repayment,
            signed_dr=signed_dr,
            verified=verified,
            bill=bill,
            der_mandatory=der[o.name][0],
            der_optional=der[o.name][1],
            uc_mandatory=uc[o.name][0],
            uc_optional=uc[o.name][1],
        )
        close(o.demand - served - native_lost - (o.repay - repayment))
        close(verified, max(0.0, o.baseline - (served + native_lost)))
    verified_total = sum(c["verified"] for c in consumers.values())
    paid = min(supply.dr_cap, verified_total)
    for c in consumers.values():
        c["paid"] = c["verified"] * paid / verified_total if verified_total else 0.0
        c["dr_payment"] = c["paid"] * supply.dr_price * DT
    backup_total = sum(c["backup"] for c in consumers.values())
    pg = pg_base + backup_total
    der_sold = sum(c["der"] for c in consumers.values())
    discarded = supply.der_generation - supply.der_charge - der_sold
    power_residual = (
        pg
        + supply.der_generation
        + discharge
        - discarded
        - supply.der_charge
        - supply.pg_charge
        - sum(c["served"] for c in consumers.values())
    )
    close(power_residual)
    assert pg <= supply.pg_max + 1e-9
    revenue = sum(c["bill"] for c in consumers.values())
    payment = paid * supply.dr_price * DT
    cash = revenue - pg * supply.pg_price * DT - backup_total * supply.backup_extra * DT - payment
    benefit = supply.dr_beta * (paid / supply.dr_reference) ** 2
    return dict(
        consumers=consumers,
        pg=pg,
        discharge=discharge,
        der_curtail=discarded,
        verified=verified_total,
        paid=paid,
        signed_dr=sum(c["signed_dr"] for c in consumers.values()),
        revenue=revenue,
        dr_payment=payment,
        cash=cash,
        benefit=benefit,
        reward_before_service_and_other_costs=cash + benefit,
        power_residual=power_residual,
    )


def main():
    cases = []

    def record(case_id, description, data, kind="counterexample_confirmed"):
        cases.append(dict(id=case_id, description=description, kind=kind, data=data))

    # C01: the narrowly stated v5 channel-switch counterexample is repaired.
    o = [Order("A", 4, uc_bid=0.4)]
    a = run(o, Supply(4, 4))
    b = run(o, Supply(0, 4, backup_contract=4))
    close(a["cash"], -0.25)
    close(b["cash"], -0.55)
    record(
        "C01",
        "Same bid and same delivered energy: changing to backup costs UC more",
        dict(ordinary=a, backup=b),
        "repair_confirmed",
    )

    # C02: the same consumer lowers its bid and receives the same backup energy.
    # Total demand is scarce (12 kW vs 8 kW); C is a valid optional repayment.
    o = [Order("A", 4, uc_bid=0.95), Order("B", 4, uc_bid=0.90), Order("C", 0, repay=4, uc_bid=0.85)]
    s = Supply(4, 8, backup_contract=4)
    a = run(o, s)
    b = run([replace(o[0], uc_bid=0.8), *o[1:]], s)
    close(a["consumers"]["A"]["served"], 4)
    close(b["consumers"]["A"]["served"], 4)
    close(a["consumers"]["A"]["bill"] - b["consumers"]["A"]["bill"], 0.15)
    close(sum(x["repay_unfilled"] for x in b["consumers"].values()), 4)
    record(
        "C02",
        "Finite backup and total scarcity still allow a profitable bid reduction",
        dict(high_bid=a, low_bid=b),
    )

    # C03: a reachable supply failure makes next-period mandatory repayment infeasible.
    queue = []
    trace = []
    for t in range(4):
        due_kwh = sum(e for deadline, e in queue if deadline <= t)
        minimum = due_kwh / DT
        local_maximum = min(4.0, sum(e for _, e in queue) / DT)
        feasible = minimum <= local_maximum
        trace.append(
            dict(
                t=t,
                queue=list(queue),
                due_kwh=due_kwh,
                repay_min_kw=minimum,
                repay_max_kw=local_maximum,
                feasible=feasible,
            )
        )
        if t < 2:
            queue.append((t + 2, 1.0))  # legal 4-kW shifting, H=2, adequate B_max
        # At t=2 a legal 4-kW repayment request receives zero electricity.
    assert trace[2]["feasible"] and not trace[3]["feasible"]
    close(trace[3]["repay_min_kw"], 8)
    record("C03", "Valid actions plus a supply failure lead to an empty action domain", trace)

    # C04: gross positive individual response is not aggregate net reduction.
    o = [Order("A", 4, curtail=4), Order("B", 4, repay=4)]
    x = run(o, Supply(8, 8, dr_cap=4))
    close(x["verified"], 4)
    close(x["signed_dr"], 0)
    close(x["pg"], 8)
    record("C04", "One consumer curtails while another repays: rewarded DR but zero net reduction", x)

    # C05: same consumer action; UC withholds optional repayment and gains reward.
    o = [Order("A", 8, curtail=4, repay=4, uc_bid=0.8)]
    a = run(o, Supply(8, 8, pg_price=0.5, dr_cap=4))
    b = run(o, Supply(4, 8, pg_price=0.5, dr_cap=4))
    close(a["verified"], 0)
    close(b["verified"], 4)
    close(a["reward_before_service_and_other_costs"], 0.6)
    close(b["reward_before_service_and_other_costs"], 0.9)
    close(b["consumers"]["A"]["native_lost"], 0)
    record(
        "C05",
        "Withheld non-due repayment increases verified DR and UC one-step reward",
        dict(
            full_service=a,
            withheld_repayment=b,
            assumptions="Not a complete dynamic best response: future/terminal costs omitted; due=0",
        ),
    )

    # C06: only the paid label changes, with physical actions unchanged.
    o = [Order("A", 10, curtail=3), Order("B", 10, curtail=3)]
    a = run(o, Supply(14, 20, dr_cap=0, dr_price=0.5))
    b = run(o, Supply(14, 20, dr_cap=4, dr_price=0.5))
    close(a["pg"], b["pg"])
    close(b["cash"] - a["cash"], -0.5)
    close(b["reward_before_service_and_other_costs"] - a["reward_before_service_and_other_costs"], 0.3)
    record(
        "C06",
        "Buying identical pre-existing response improves utility but worsens cash",
        dict(no_purchase=a, purchase=b),
    )

    # C07: priority is local to each resource pool, not system-wide.
    o = [Order("A", 0, repay=4, der_share=1), Order("B", 4, der_share=0)]
    x = run(o, Supply(0, 0, der_generation=4))
    close(x["consumers"]["A"]["repay_served"], 4)
    close(x["consumers"]["B"]["native_lost"], 4)
    record("C07", "Optional DER use can coexist with another consumer native shortage", x)

    # C08: reproduce the supplied v7 worked example exactly.
    o = [
        Order("1", 6, curtail=1, defer=1, der_share=0.5, uc_bid=0.9, der_bid=0.7),
        Order("2", 5, curtail=1, der_share=0.5, uc_bid=0.85, der_bid=0.65),
        Order("3", 4, repay=1, der_share=0.4, uc_bid=0.8, der_bid=0.6),
    ]
    x = run(o, Supply(6, 8, der_generation=4, backup_contract=2, dr_cap=2))
    close(x["revenue"], 2.35)
    close(x["cash"], 0.8)
    close(x["benefit"], 0.2)
    close(x["reward_before_service_and_other_costs"], 1)
    record(
        "C08",
        "Supplied v7 example is arithmetically consistent under its stated omissions",
        x,
        "example_confirmed",
    )

    # C09: positive quadratic utility produces a threshold and endpoint maximizers.
    grid = [k / 100 for k in range(401)]

    def value(q, beta):
        return beta * (q / 4) ** 2 - 0.2 * q * DT

    low = max(grid, key=lambda q: value(q, 0.1))
    high = max(grid, key=lambda q: value(q, 0.8))
    close(low, 0)
    close(high, 4)
    close(value(2, 0.8), 0.1)
    record(
        "C09",
        "Positive DR value coefficient does not ensure positive procurement",
        dict(
            low_beta=0.1,
            low_argmax=low,
            high_beta=0.8,
            high_argmax=high,
            endpoint_threshold_beta=0.2,
            derivative_at_zero=-0.2 * DT,
        ),
    )

    # C10: a finite shortage penalty does not preclude strategic non-service.
    ordinary = (0.4 - 0.65) * 1
    not_served = -0.1 * 1
    assert not_served > ordinary
    record(
        "C10",
        "Small native-service penalty leaves profitable UC non-service",
        dict(ordinary_reward=ordinary, unserved_reward=not_served, penalty=0.1),
    )

    # C11: the supplied source's claimed quotation differs from PDF equation (10).
    q, m, ts = 2.0, 4.0, 0.8
    record(
        "C11",
        "PDF equation (10) has denominator m, not m squared",
        dict(pdf_benefit=ts * q * q / m, supplied_v7_quote=ts * (q / m) ** 2),
        "source_mismatch_confirmed",
    )

    # C12: terminal stock is economically material even with zero DR.
    discharged_kwh, price, degradation = 1.0, 0.4, 0.01
    record(
        "C12",
        "Resetting gifted ESS stock rewards selling it at the episode end",
        dict(
            initial_stock_kwh=discharged_kwh,
            residual_stock_kwh=0,
            cash_receipt=0.4,
            degradation=0.01,
            objective_gain_without_terminal_stock_value=(price - degradation) * discharged_kwh,
        ),
        "boundary_condition_example",
    )

    # C13: equal disclosed UC capacity and reserve contract hide supply composition.
    o = [Order("A", 4), Order("B", 4)]
    a = run(o, Supply(0, 4, discharge=4, backup_contract=4))
    b = run(o, Supply(4, 4, discharge=0, backup_contract=4))
    close(sum(c["served"] for c in a["consumers"].values()), 8)
    close(sum(c["served"] for c in b["consumers"].values()), 4)
    record(
        "C13",
        "Same proposed consumer observation hides different backup capability",
        dict(shared_public_uc_capacity=4, shared_backup_contract=4, ess_supported=a, pg_supported=b),
    )

    # C14: protecting native load requires constraining defer as well as curtail.
    baseline, hard_native, maximum, curtail = 4.0, 2.0, 4.0, 0.0
    curtail_max = min(baseline - hard_native, maximum)
    defer_max = min(baseline - curtail, maximum - curtail, 4 / DT)
    assert curtail <= curtail_max and baseline - curtail - defer_max < hard_native
    record(
        "C14",
        "Optional hard-native-load rule constrains curtailment but not deferral",
        dict(
            baseline=baseline,
            hard_native=hard_native,
            curtail=curtail,
            defer_allowed_by_equation_7=defer_max,
            remaining_native=baseline - curtail - defer_max,
        ),
    )

    rng = random.Random(7102026)
    max_power_residual = max_cash_residual = max_queue_residual = 0.0
    for _ in range(5000):
        pg_max = rng.uniform(1, 60)
        generation = rng.uniform(0, 30)
        charge_mode = rng.choice((True, False))
        der_charge = rng.uniform(0, generation) if charge_mode else 0
        pg_plan = rng.uniform(0, pg_max)
        pg_charge = rng.uniform(0, pg_plan) if charge_mode else 0
        discharge = 0 if charge_mode else rng.uniform(0, 20)
        supply = Supply(
            pg_plan,
            pg_max,
            generation,
            rng.uniform(0, generation - der_charge),
            der_charge,
            pg_charge,
            discharge,
            rng.uniform(0, pg_max),
            pg_price=rng.uniform(0.1, 1),
            dr_cap=rng.uniform(0, 20),
            dr_price=rng.uniform(0, 0.5),
        )
        orders = []
        backlogs = {}
        for k in range(rng.randint(1, 8)):
            baseline = rng.uniform(0, 15)
            maximum = rng.uniform(0, 10)
            backlog = rng.uniform(0, 6)
            backlogs[str(k)] = backlog
            curtail = rng.uniform(0, min(baseline, maximum))
            if rng.choice((True, False)):
                defer = rng.uniform(0, min(baseline - curtail, maximum - curtail, (6 - backlog) / DT))
                repay = due = 0
            else:
                defer = 0
                repay = rng.uniform(0, min(maximum - curtail, backlog / DT))
                due = rng.uniform(0, repay)
            orders.append(
                Order(
                    str(k),
                    baseline,
                    curtail,
                    defer,
                    repay,
                    due,
                    rng.choice((0, 0.25, 0.5, 1)),
                    rng.choice((0.4, 0.6, 0.8, 1)),
                    rng.choice((0.3, 0.5, 0.7)),
                )
            )
        x = run(orders, supply)
        max_power_residual = max(max_power_residual, abs(x["power_residual"]))
        cash_residual = (
            sum(c["bill"] - c["dr_payment"] for c in x["consumers"].values())
            - x["pg"] * supply.pg_price * DT
            - sum(c["backup"] for c in x["consumers"].values()) * supply.backup_extra * DT
            - x["cash"]
        )
        close(cash_residual)
        max_cash_residual = max(max_cash_residual, abs(cash_residual))
        for o in orders:
            c = x["consumers"][o.name]
            closing = backlogs[o.name] + (o.defer - c["repay_served"]) * DT
            assert -1e-9 <= closing <= 6 + 1e-9
            # Construct explicit task balances independently of scalar update.
            old_remaining = max(0, backlogs[o.name] - c["repay_served"] * DT)
            explicit_sum = old_remaining + o.defer * DT
            residual = explicit_sum - closing
            close(residual)
            max_queue_residual = max(max_queue_residual, abs(residual))
        # A feasible planned ESS state remains bounded after reducing discharge.
        eta_c, eta_o = 0.95, 0.92
        capacity = 40.0
        initial = discharge * DT / eta_o if discharge else 0.0
        planned = initial + eta_c * (der_charge + pg_charge) * DT - discharge * DT / eta_o
        actual = initial + eta_c * (der_charge + pg_charge) * DT - x["discharge"] * DT / eta_o
        assert 0 <= planned <= capacity and -1e-9 <= actual <= capacity

    inputs = {
        "specified_v7": Path(r"C:\Users\ykq31\Downloads\Stackelberg_Nash_Dual_PAB_DR_Modeling_ZH_v7.md"),
        "workspace_v5": ROOT / "Stackelberg_Nash_Pay_as_Bid_Modeling_ZH_v5.md",
        "paper": Path(
            r"D:\Yang\论文\论文\Asynchronous_Multi-Agent_Reinforcement_Learning-Based_Framework_for_Bi-Level_Noncooperative_Game-Theoretic_Demand_Response(1).pdf"
        ),
    }
    manifest = {
        k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k, p in inputs.items()
    }
    report = dict(
        date="2026-10-10",
        source_manifest=manifest,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        meaning="Assertions passed means described result reproduced; this is not environment or equilibrium certification.",
        limitations=[
            "Independent static formula checker, not current snmarket v5 tests",
            "No trained policies, full best responses, formal equilibrium proof, or forecast uncertainty",
        ],
        cases=cases,
        random_static_checks=dict(
            samples=5000,
            seed=7102026,
            max_power_residual=max_power_residual,
            max_cash_residual=max_cash_residual,
            max_queue_residual=max_queue_residual,
        ),
        status="all_assertions_passed",
    )
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(cases=len(cases), samples=5000, status=report["status"], output=str(OUT)), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
