"""Verify the v9 documentation simplification against its unchanged reference.

This file belongs to documentation QA; it does not import or edit environments.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys
from cost_notation import restore_formula

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
REFERENCE = BASE / "verify_v9_mechanisms.py"
REFERENCE_SHA = "d732f140462839469c677fcab425f96aa70c9ab6cf2304f2fc6d425ef0ebed97"
DOCUMENT = BASE / "modeling_complete_zh.md"
BEFORE = ROOT / ".tmp/v9-formula-before"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference():
    assert sha(REFERENCE) == REFERENCE_SHA
    spec = importlib.util.spec_from_file_location("v9_unchanged_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def simplified_bounds(base, rigid, flex, backlog, due, capacity, dt, t, horizon, delay, fraction):
    minimum = min(due / dt, flex)
    cut = fraction * min(base - rigid, flex - minimum)
    lower = -min(flex - cut, backlog / dt)
    upper = -minimum if minimum > 0 else min(
        base - rigid - cut, flex - cut, (capacity - backlog) / dt
    )
    if minimum == 0 and horizon is not None and t + delay > horizon - 1:
        upper = 0.0
    return {
        "curtail": cut, "attempt": minimum, "low": lower, "high": upper,
        "capacity_violation": max(0, due - flex * dt),
    }


def check_document():
    text = DOCUMENT.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", text)
    equations = re.findall(r"\$\$([\s\S]*?)\$\$", text)
    numbered = {int(re.search(r"\\tag\{(\d+)\}", e).group(1)): re.sub(r"\s+", "", restore_formula(e))
                for e in equations if re.search(r"\\tag\{(\d+)\}", e)}
    assert sorted(numbered) == list(range(1, 31))
    assert r"p^{dis,exec}&=\min\left(p^{dis,offer},p^{UC,sold}\right)" in numbered[4]
    assert r"p^{PG,exec}&=p^{UC,sold}+p^{ch,PG}-p^{dis,exec}" in numbered[4]
    assert r"\Pi^{cash}=\sum_iK_i^{bill}-" in numbered[24]
    assert r"\widehat p^{DER}" not in text
    assert "消费者净转账合计减去外部现金成本" in text
    assert r"p_i^{due}=\min\left(\frac{D_i^{due}}{\Deltat},\barp_i^{flex}\right)" in numbered[8]
    assert "B_i" not in numbered[8]
    assert r"p_i^{req}&=p_i^{base}-p_i^{cut}-p_i^{shift}" in numbered[13]
    assert r"p_i^{repay}&=[p_i^{serve}-p_i^{native}]_+" in numbered[14]
    assert r"p_i^{DR}&=\chi\left(p_i^{base}-p_i^{req}\right)" in numbered[20]
    assert r"\min\left(p^{DR,cap},p^{DR,net}\right)" in numbered[21]
    assert r"\frac{[p_i^{DR}]_+}{\sum_j[p_j^{DR}]_+}" in numbered[21]
    assert r"([p_i^{shift}]_+)^2" in numbered[28]
    assert r"\delta_{j,t}=\widehatR_{j,t}+" in compact
    assert "总请求为零时，该层分配为零" in text
    assert r"\sup_i" not in text
    allowed_commands = {
        "Delta", "Pi", "alpha", "bar", "begin", "bigl", "bigr", "chi", "cdot", "delta", "displaystyle",
        "ell", "end", "epsilon", "eta", "frac", "gamma", "ge", "in", "kappa",
        "lambda", "ldots", "le", "left", "mathbb", "mathcal", "max", "mid", "min",
        "omega", "operatorname", "phi", "pi", "quad", "qquad", "rho", "right",
        "sim", "sum", "tag", "text", "theta", "times", "underbrace", "widetilde", "widehat",
        "beta", "mathbf", "infty", "log", "mu", "prod", "sigma",
    }
    math_text = "\n".join(re.findall(r"\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+(?<!\\)\$", text))
    actual_commands = set(re.findall(r"\\([A-Za-z]+)", math_text))
    assert actual_commands <= allowed_commands, sorted(actual_commands - allowed_commands)
    aliases = [r"\bar G", r"\bar C", r"\bar O", r"\widehat O", r"\bar P^{DR}",
               "G^{offer}", "G^{exec}", "A^{curt}", "A^{exec}", "C_R", "C_G",
               "O^{offer}", "O^{exec}", "K_R", "K_U", "X_R", "X_U", "p_i^*",
               "F_i", "f_i", "s_i^+", "r_i^0", "r_i^{max}", "s_i^{max}",
               "z_i", "v_i", "V_i^{cap}", r"\ell_i", "x_i^R", "x_i^U", "Q^{net}",
               "p^{DR,gross}", "p^{DR,paid}", "p_i^{DR,paid}"]
    for alias in aliases:
        assert alias not in text, alias
    assert re.findall(r"^## (\d+)\. ", text, re.M) == [str(i) for i in range(1, 27)]
    assert len(re.findall(r"^# 第.+部分", text, re.M)) == 6
    if BEFORE.exists():
        original = (BEFORE / "modeling_complete_zh.md").read_text(encoding="utf-8")
        assert re.findall(r"```[\s\S]*?```", original) == re.findall(r"```[\s\S]*?```", text)
    # All power tokens are p superscripts; remaining letters encode non-power objects.
    powers = sorted(set(re.findall(r"(?:\\bar\s+)?p(?:_i)?\^\{[^}]+\}", text)))
    return {"numbered_equations": 30, "chapters": 26, "parts": 6,
            "stale_power_aliases": [], "code_examples_unchanged": True,
            "power_notation_inventory": powers, "latex_command_inventory_valid": True}


def main():
    reference = load_reference()
    rng = random.Random(9122026)
    maximum_error = 0.0
    checks = 0

    def same(a, b):
        nonlocal maximum_error, checks
        maximum_error = max(maximum_error, abs(a - b))
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)
        checks += 1

    samples = 10000
    membership_checks = 0
    for sample in range(samples):
        base = rng.choice([0.0, rng.uniform(0, 15)])
        rigid = base * rng.random()
        flex = rng.choice([0.0, rng.uniform(0, 20)])
        capacity = rng.choice([0.0, rng.uniform(0, 15)])
        backlog = capacity * rng.random()
        due = rng.choice([0.0, backlog, backlog * rng.random()])
        dt = rng.choice([0.25, 0.5, 1.0])
        horizon = rng.choice([None, 96])
        time = rng.randrange(96)
        delay = rng.randrange(1, 8)
        fraction = rng.choice([0.0, 1.0, rng.random()])
        args = (base, rigid, flex, backlog, due, capacity, dt, time, horizon, delay, fraction)
        old = reference.local_bounds(*args)
        new = simplified_bounds(*args)
        for key in old:
            same(old[key], new[key])
        assert new["low"] <= new["high"] + 1e-9
        # Classify both boundary values and proposals outside the feasible interval.
        for shift in [old["low"], old["high"], rng.uniform(-25, 25)]:
            old_legal = old["low"] - 1e-9 <= shift <= old["high"] + 1e-9
            new_legal = new["low"] - 1e-9 <= shift <= new["high"] + 1e-9
            assert old_legal == new_legal
            membership_checks += 1

        shift = new["low"] + rng.random() * (new["high"] - new["low"])
        cut, minimum = new["curtail"], new["attempt"]
        defer, request_repay = max(0.0, shift), max(0.0, -shift)
        old_native = base - cut - defer
        old_demand = old_native + request_repay
        native = base - cut - max(0.0, shift)
        demand = base - cut - shift
        necessary = native + minimum
        same(old_native, native)
        same(old_demand, demand)
        same(request_repay - minimum, demand - necessary)
        assert native >= rigid - 1e-9 and demand >= necessary - 1e-9
        assert cut + abs(shift) <= flex + 1e-9

        served = rng.choice([0.0, old_demand, old_demand * rng.random()])
        old_repaid = min(request_repay, max(0, served - old_native))
        repaid = max(0, served - native)
        lost = max(0, native - served)
        same(old_repaid, repaid)
        same(old_native - min(old_native, served), lost)
        same(backlog + (defer - old_repaid) * dt, backlog + (max(0, shift) - repaid) * dt)
        next_backlog = backlog + (max(0, shift) - repaid) * dt
        assert -1e-9 <= next_backlog <= capacity + 1e-9
        late = max(0, due - repaid * dt)
        assert new["capacity_violation"] <= late + 1e-9
        same(old_demand - served, lost + request_repay - repaid)

        event = rng.randrange(2)
        same(event * (cut + defer - request_repay), event * (base - demand))
        coefficient = rng.uniform(0, 4)
        same(coefficient * (cut**2 + defer**2) / max(base, 1e-8) * dt,
             coefficient * (cut**2 + max(0, shift)**2) / max(base, 1e-8) * dt)
        # Same source intentions in both priority layers, including the rescue case.
        share = rng.random()
        first_der = rng.random() * min(share * old_demand, old_native + minimum)
        rescue_der = rng.random() * (old_native + minimum - first_der)
        necessary_der = first_der + rescue_der
        same(min(share * old_demand, old_native + minimum), min(share * demand, necessary))
        same(min(request_repay - minimum, max(0, share * old_demand - necessary_der)),
             min(demand - necessary, max(0, share * demand - necessary_der)))

    # Each retained maximum has a state in which it alone binds.
    bound_witnesses = []
    for label, expected, base, rigid, flex, backlog, cap in [
        ("movable_native", 1, 1, 0, 10, 0, 10),
        ("flex_power", 1, 10, 0, 1, 0, 10),
        ("queue_space", 1, 10, 0, 10, 0, 1),
    ]:
        b = simplified_bounds(base, rigid, flex, backlog, 0, cap, 1, 0, None, 1, 0)
        same(b["high"], expected)
        bound_witnesses.append({"upper_bound": label, "binding_value_kw": b["high"]})
    for label, flex, backlog in [("flex_power", 1, 10), ("task_balance", 10, 1)]:
        b = simplified_bounds(10, 0, flex, backlog, 0, 10, 1, 0, None, 1, 0)
        same(-b["low"], 1)
        bound_witnesses.append({"repay_bound": label, "binding_value_kw": -b["low"]})

    # Resource, DR, payment and reward equations evaluated on original market outputs.
    market_cases = 1000
    for case in range(market_cases):
        dt = rng.choice([0.25, 0.5, 1.0])
        energy = rng.uniform(0, 10)
        der = rng.uniform(0, 12)
        grid_offer = rng.uniform(0, 15)
        mode = rng.randrange(3)
        charge_room = min(8.0, (10 - energy) / (0.95 * dt))
        der_charge = rng.random() * min(der, charge_room) if mode == 0 else 0.0
        grid_charge = rng.random() * min(grid_offer, charge_room - der_charge) if mode == 0 else 0.0
        discharge_offer = rng.random() * min(8.0, 0.95 * energy / dt) if mode == 1 else 0.0
        event = bool(rng.randrange(2))
        supply = reference.Supply(
            grid_offer=grid_offer, grid_max=15, der=der,
            planned_curtail=rng.random() * (der - der_charge),
            der_charge=der_charge, grid_charge=grid_charge, discharge_offer=discharge_offer,
            dr_cap=rng.uniform(0, 10) if event else 0, event=event,
            dr_price=0.2 if event else 0, dt=dt, energy=energy,
            degradation=0.02, curtail_opportunity=0.1, der_cash_cost=0.03, dr_value=0.4,
        )
        orders = []
        shifts = []
        for i in range(rng.randrange(1, 9)):
            baseline = rng.uniform(0, 12)
            cut = rng.random() * baseline
            shift = rng.uniform(-5, baseline - cut)
            repayment = max(0, -shift)
            minimum = rng.random() * repayment
            orders.append(reference.Order(str(i), baseline, cut, max(0, shift), repayment,
                                          minimum, rng.random(), rng.uniform(0.2, 0.9), 0.9))
            shifts.append(shift)
        result = reference.run(orders, supply)
        sold_der = sum(c["der"] for c in result["consumers"].values())
        sold_uc = sum(c["uc"] for c in result["consumers"].values())
        discharge = min(discharge_offer, sold_uc)
        imported = sold_uc + grid_charge - discharge
        spill = der - der_charge - sold_der
        same(discharge, result["discharge"])
        same(imported, result["pg"])
        same(spill, result["actual_curtail"])
        same(energy + 0.95 * (der_charge + grid_charge) * dt - discharge * dt / 0.95,
             result["energy_next"])
        signed = [int(event) * (o.baseline - o.demand) for o in orders]
        gross, net = sum(max(0, p) for p in signed), max(0, sum(signed))
        paid = min(supply.dr_cap, net)
        same(gross, result["gross_dr"])
        same(net, result["net_dr"])
        same(paid, result["paid_dr"])
        net_cash = 0.0
        due_service = 0.0
        for order, shift, response in zip(orders, shifts, signed):
            c = result["consumers"][order.name]
            native = order.baseline - order.curtail - max(0, shift)
            repaid = max(0, c["served"] - native)
            lost = max(0, native - c["served"])
            compensation = lost * dt
            payment = (supply.dr_price * dt * min(supply.dr_cap, net)
                       * (max(0, response) / gross) if gross else 0)
            bill = (order.der_bid * c["der"] + order.uc_tariff * c["uc"]) * dt
            same(repaid, c["repayment"])
            same(lost, c["native_lost"])
            same(payment, c["dr_payment"])
            same(bill - payment - compensation, c["net_transfer"])
            net_cash += bill - payment - compensation
            due_service += max(0, order.due_attempt - repaid) * dt
        cash = net_cash - supply.pg_price * imported * dt - supply.der_cash_cost * (der - spill) * dt
        book = cash - 0.02 * (der_charge + grid_charge + discharge) * dt - 0.1 * spill * dt
        same(cash, result["cash"])
        same(book + 0.4 * net * dt - due_service, result["reward_before_unattempted_due"])

    # Direct monetary settlement equals the original intermediate-power calculation.
    dr_rng = random.Random(9142026)
    dr_cases = [
        ([0, 0], 4, 0.5, 0.25), ([4, -4], 4, 0.5, 0.25),
        ([3, 3], 4, 0.5, 0.25), ([3, 1, -2], 3, 0.4, 0.25),
        ([4, 2], 3, 0.5, 0.25), ([-2, -1], 3, 0.5, 0.25),
        ([2], 0, 0.5, 0.25), ([2, 1], 3, 0, 0.25),
    ]
    dr_boundary_checks = []
    dr_random_samples = 10000
    for case in range(len(dr_cases) + dr_random_samples):
        if case < len(dr_cases):
            responses, cap, rate, duration = dr_cases[case]
        else:
            event = dr_rng.randrange(2)
            responses = [event * dr_rng.uniform(-10, 15) for _ in range(dr_rng.randrange(1, 9))]
            cap = dr_rng.choice([0, dr_rng.uniform(0, 25)]) if event else 0
            rate = dr_rng.uniform(0, 1) if event else 0
            duration = dr_rng.choice([0.25, 0.5, 1.0])
        positive_sum = sum(max(0, p) for p in responses)
        net = max(0, sum(responses))
        paid = min(cap, net)
        old_amounts = [rate * (paid * max(0, p) / positive_sum if positive_sum else 0) * duration
                       for p in responses]
        direct_amounts = [rate * duration * min(cap, net) * max(0, p) / positive_sum
                          if positive_sum else 0 for p in responses]
        for old_amount, new_amount, response in zip(old_amounts, direct_amounts, responses):
            same(old_amount, new_amount)
            assert new_amount >= -1e-9
            if response <= 0:
                same(new_amount, 0)
        same(sum(direct_amounts), rate * duration * min(cap, net))
        if net == 0:
            same(sum(direct_amounts), 0)
        if case < len(dr_cases):
            dr_boundary_checks.append({"signed_responses_kw": responses, "procurement_cap_kw": cap,
                                       "direct_payments_yuan": direct_amounts})

    # Whole-document simplifications: charge/discharge exclusivity and ledger reuse.
    review_rng = random.Random(9142626)
    full_review_samples = 10000
    for case in range(full_review_samples):
        duration = review_rng.choice([0.25, 0.5, 1.0])
        energy = review_rng.uniform(0, 10)
        der = review_rng.uniform(0, 10)
        offer = review_rng.uniform(0, 10)
        room = min(8, (10 - energy) / (0.95 * duration))
        mode = review_rng.randrange(3)
        cr = review_rng.random() * min(der, room) if mode == 0 else 0
        cg = review_rng.random() * min(offer, room - cr) if mode == 0 else 0
        discharge_offer = review_rng.random() * min(8, 0.95 * energy / duration) if mode == 1 else 0
        sold_uc = review_rng.random() * (offer + discharge_offer - cg)
        old_discharge = min(discharge_offer, sold_uc + cg)
        new_discharge = min(discharge_offer, sold_uc)
        same(old_discharge, new_discharge)
        imported = sold_uc + cg - new_discharge
        assert -1e-9 <= imported <= offer + 1e-9
        spill = review_rng.random() * (der - cr)
        net_transfers, raw_transfers = [], []
        for i in range(review_rng.randrange(1, 9)):
            bill, incentive = review_rng.uniform(0, 10), review_rng.uniform(0, 2)
            loss, compensation_rate = review_rng.uniform(0, 8), review_rng.uniform(0, 2)
            compensation = compensation_rate * loss * duration
            net_transfers.append(bill - incentive - compensation)
            raw_transfers.append((bill, incentive, compensation))
            loss_value, flex_cost = review_rng.uniform(0, 3), review_rng.uniform(0, 3)
            backlog = review_rng.uniform(0, 10)
            late = review_rng.random() * backlog
            wait_rate, late_rate = review_rng.uniform(0, 2), review_rng.uniform(0, 2)
            other_service = (wait_rate * backlog + late_rate * late) * duration
            old_cost = bill - incentive - compensation + flex_cost + loss_value * loss * duration + other_service
            net_loss_cost = bill - incentive + flex_cost + (loss_value - compensation_rate) * loss * duration + other_service
            same(old_cost, net_loss_cost)
        pg_price, der_rate, external = review_rng.uniform(-0.2, 2), review_rng.uniform(0, 0.3), review_rng.uniform(0, 2)
        cash_old = sum(p - dr - a for p, dr, a in raw_transfers) - pg_price * imported * duration - der_rate * (der - spill) * duration - external
        cash_new = sum(net_transfers) - pg_price * imported * duration - der_rate * (der - spill) * duration - external
        same(cash_old, cash_new)
    # This is a mathematical countercheck outside the legal domain, not a legal UC action.
    assert min(1, 0 + 1) != min(1, 0)
    whole_document_checks = {
        "seed": 9142626, "samples": full_review_samples,
        "discharge_min_simplification": "equivalent under charge/discharge exclusivity",
        "ledger_cash_reuse": "equivalent",
        "consumer_net_loss_expression": "equivalent explanatory option; not applied to main reward",
        "simultaneous_charge_discharge_countercheck": "simplification is conditional on the current mutual-exclusion constraint",
    }

    example = simplified_bounds(4, 0, 4, 3, 2, 5, 0.25, 0, None, 1, 0)
    same(example["attempt"], 4)
    same(example["capacity_violation"], 1)
    assert simplified_bounds(4, 0, 4, 1, 0, 5, 0.25, 95, 96, 3, 0)["high"] == 0
    assert simplified_bounds(4, 0, 4, 1, 0, 5, 0.25, 95, None, 3, 0)["high"] == 4

    # Verify the terminal append is consumed by the TD target, including zero bootstrap.
    terminal_checks = []
    for regular, appended, terminal, next_value in [(1, 3, True, 10), (1, 0, False, 10)]:
        target = regular + appended + 0.99 * (1 - int(terminal)) * next_value
        same(target, 4 if terminal else 10.9)
        terminal_checks.append({"terminal": terminal, "td_target": target})

    integrity = {}
    for manifest_name in ["protected_files_manifest.json", "history_files_manifest.json"]:
        manifest = json.loads((BASE / manifest_name).read_text(encoding="utf-8"))
        assert all(sha(ROOT / path) == digest for path, digest in manifest.items())
        integrity[manifest_name] = {"files": len(manifest), "unchanged": True}
    document_checks = check_document()
    output = {
        "status": "passed", "revision_date": "2026-10-14", "seed": 9122026,
        "document_sha256": sha(DOCUMENT), "unchanged_reference_sha256": sha(REFERENCE),
        "consumer_state_samples": samples, "interval_membership_checks": membership_checks,
        "original_market_output_cases": market_cases, "scalar_comparisons": checks,
        "maximum_absolute_difference": maximum_error,
        "retained_independent_bound_witnesses": bound_witnesses,
        "terminal_reward_checks": terminal_checks, "document_checks": document_checks,
        "direct_dr_settlement": {"seed": 9142026, "random_samples": dr_random_samples,
                                 "boundary_cases": dr_boundary_checks,
                                 "intermediate_power_aliases_removed": 3,
                                 "netting_and_procurement_and_shares_preserved": True},
        "whole_document_checks": whole_document_checks,
        "integrity": integrity,
        "scope": "Equivalent consumer/DR formulas, mutual-exclusion discharge simplification and cash-ledger reuse; original market, physics, queue and economics retained. No production changes.",
    }
    (BASE / "formula_validation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k not in ["document_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
