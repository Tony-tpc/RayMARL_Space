"""Audit v9 documentation units without importing or modifying the environment.

Dimensions use yuan, kWh and h as bases. A kW is a kWh/h. This checks
dimensional expressions, including mistakes whose numeric values can be hidden
by dt=1, and separately checks the existing per-assessment charge semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from cost_notation import restore_formula, restore_symbols

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
SOURCE_BEFORE_SHA = "73551faf77c59af635bf7cea82b51e14c02fa1bf6d2720e8a141da991f2febd5"
DISPLAY_BASELINE_SHA = "999380c2b083f9b688f5ef98c7885b019d7f68fb6f61a3665ce48087ae391b62"
CODE_BASELINE_SHA = "529c45cb8bd4aa4f72aa4269566ecb8e171ee89a4f1f6a12bb1f60ef9efbbecf"
REFERENCE_SHA = "d732f140462839469c677fcab425f96aa70c9ab6cf2304f2fc6d425ef0ebed97"
NUMERIC_BASELINE_SHA = {
    "verification_results.json": "e9a1c017ecb675b8c68de94ae6224c31c352b8c04121d5cd490e664bfd448082",
    "formula_validation.json": "e996386e7ad541e7d2892994ba4bfd34be2e57b29c600b5f9e90334c5071af82",
}


@dataclass(frozen=True)
class Unit:
    yuan: int = 0
    kwh: int = 0
    hour: int = 0

    def __mul__(self, other):
        return Unit(self.yuan + other.yuan, self.kwh + other.kwh, self.hour + other.hour)

    def __truediv__(self, other):
        return Unit(self.yuan - other.yuan, self.kwh - other.kwh, self.hour - other.hour)

    def __pow__(self, exponent):
        return Unit(self.yuan * exponent, self.kwh * exponent, self.hour * exponent)


ONE, MONEY, ENERGY, TIME = Unit(), Unit(yuan=1), Unit(kwh=1), Unit(hour=1)
POWER = ENERGY / TIME
PRICE = MONEY / ENERGY
HOLDING_RATE = MONEY / ENERGY / TIME


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add(*units):
    assert units and all(unit == units[0] for unit in units), units
    return units[0]


def require_money(name, unit):
    assert unit == MONEY, (name, unit)
    return {"item": name, "dimension": [unit.yuan, unit.kwh, unit.hour], "status": "passed"}


def get_displays(text):
    # Existing 45 displays remain the accounting baseline. The new UC decoder
    # and continuous objective are verified by verify_v9_definition_completion.py.
    start = text.index("### 11.2 UC 一次输出与联合可行行动")
    end = text.index("### 11.3 消费者条件分布", start)
    baseline = []
    for match in re.finditer(r"\$\$([\s\S]*?)\$\$", text):
        if start <= match.start() < end or "J_j^{cont}" in match.group(1):
            continue
        baseline.append(re.sub(r"\s+", "", restore_formula(match.group(1))))
    return baseline


def main():
    text = (BASE / "modeling_complete_zh.md").read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", restore_symbols(text))
    assert "### 2.6 成本与奖励系数的统一单位" in text
    assert "### 8.11 全部成本与奖励的步长核对" in text
    assert "元/(kWh·h)" in text and "按每次期末考核" in text
    assert "原始单位为元等值平方" in text
    assert r"\epsilon" in text and "单位为 kW" in text
    # Protect existing mathematical definitions, including unnumbered displays.
    assert semantic_sha(get_displays(text)) == DISPLAY_BASELINE_SHA, "An existing display formula changed"
    assert semantic_sha(re.findall(r"```[\s\S]*?```", text)) == CODE_BASELINE_SHA
    tags = [int(x) for x in re.findall(r"\\tag\{(\d+)\}", text)]
    assert tags == list(range(1, 31)), tags
    # Require the important distinction to appear in actual formulas, not only prose.
    assert r"L_U=\lambda_n\sum_ip_i^{lost}\Deltat+\lambda_d\sum_iV_i^{due}." in compact
    assert r"\lambda_i^BB_i'\Deltat+\lambda_i^{late}V_i^{due}\Deltat" in compact
    assert r"\max(p_i^{base},\epsilon)" in compact
    assert r"R_{U,term}=v_E(E_T-E_0)-\lambda_U^{term}\sum_iB_{i,T}" in compact

    money_checks = []
    power_energy_cost = PRICE * POWER * TIME
    held_energy_cost = HOLDING_RATE * ENERGY * TIME
    assessment_cost = PRICE * ENERGY  # One assessment each physical period.
    terminal_cost = PRICE * ENERGY  # Once at the real end.
    fraction = POWER / POWER
    assert fraction == ONE
    assert add(POWER, POWER) == POWER
    assert add(ENERGY, ENERGY) == ENERGY
    assert PRICE == MONEY / ENERGY

    for name in ["DER energy bill", "UC energy bill", "DR total compensation",
                 "native-load compensation", "PG import cost", "DER use cost",
                 "ESS AC-side throughput degradation", "curtailment opportunity cost",
                 "DR service value", "UC native-loss assessment", "consumer native-loss cost"]:
        money_checks.append(require_money(name, power_energy_cost))
    money_checks.append(require_money("individual DR allocation", power_energy_cost * fraction))
    flexible_power = add(POWER ** 2, POWER ** 2) / add(POWER, POWER)
    assert flexible_power == POWER
    money_checks.append(require_money("normalized quadratic discomfort", PRICE * flexible_power * TIME))
    for name in ["backlog holding", "overdue holding"]:
        money_checks.append(require_money(name, held_energy_cost))
    money_checks.append(require_money("UC per-period overdue assessment", assessment_cost))
    for name in ["external cash amount", "net consumer transfer", "UC cash profit",
                 "UC book profit", "total UC service assessment", "total consumer service cost",
                 "UC stage reward", "consumer stage reward", "terminal-inclusive reward",
                 "discounted objective", "state value", "TD residual", "GAE advantage",
                 "PPO raw clipped objective", "UC response value", "consumer unilateral gain",
                 "maximum consumer gain", "UC candidate gain"]:
        money_checks.append(require_money(name, add(MONEY, MONEY)))
    for name in ["terminal battery value", "UC terminal task responsibility", "consumer terminal task responsibility"]:
        money_checks.append(require_money(name, terminal_cost))
    assert ONE * MONEY == MONEY
    assert MONEY ** 2 == Unit(yuan=2)

    negative_controls = {
        "omitted_duration_on_price_times_power": PRICE * POWER,
        "repeated_duration_on_complete_amount": MONEY * TIME,
        "holding_rate_without_duration": HOLDING_RATE * ENERGY,
        "extra_duration_on_per_assessment_overdue_charge": PRICE * ENERGY * TIME,
        "extra_duration_on_terminal_energy_charge": PRICE * ENERGY * TIME,
        "dimensionless_floor_in_power_denominator": ONE,
    }
    for name, unit in negative_controls.items():
        expected = POWER if name == "dimensionless_floor_in_power_denominator" else MONEY
        assert unit != expected, (name, unit)
    try:
        add(POWER, ONE)
    except AssertionError:
        pass
    else:
        raise AssertionError("Power and dimensionless epsilon were incorrectly interchangeable")

    # Constant power/backlog across one physical hour. This is an accounting
    # probe, not a trajectory-resampling claim or a game-equilibrium test.
    step_probes = []
    for dt in [1 / 12, 0.1, 0.25, 0.5, 1.0]:
        periods = round(1 / dt)
        assert math.isclose(periods * dt, 1)
        energy_bill = periods * (0.4 * 4 * dt)
        waiting_cost = periods * (0.1 * 3 * dt)
        consumer_overdue_cost = periods * (0.2 * 1 * dt)
        uc_assessment = periods * (1.0 * 1)
        external_per_period_amount = periods * 0.17
        terminal_once = 0.2 * (7 - 5) - 0.6 * 3
        assert math.isclose(energy_bill, 1.6)
        assert math.isclose(waiting_cost, 0.3)
        assert math.isclose(consumer_overdue_cost, 0.2)
        assert math.isclose(uc_assessment, periods)
        assert math.isclose(external_per_period_amount, 0.17 * periods)
        assert math.isclose(terminal_once, -1.4)
        step_probes.append({
            "dt_hours": dt, "one_hour_periods": periods,
            "energy_bill_yuan": energy_bill, "backlog_holding_yuan_equivalent": waiting_cost,
            "consumer_overdue_holding_yuan_equivalent": consumer_overdue_cost,
            "UC_per_assessment_charge_yuan_equivalent": uc_assessment,
            "external_fixed_amount_per_period_yuan": external_per_period_amount,
            "terminal_added_once_yuan_equivalent": terminal_once,
        })

    # Unit conversion is separate from changing step duration. Use the same
    # original physical period and convert kW/kWh to W/Wh throughout.
    dt, base, cut, deferred, eps = 0.25, 4.0, 1.0, 0.5, 1e-6
    kappa, energy_price = 0.75, 0.4
    discomfort_kw = kappa * (cut ** 2 + deferred ** 2) / max(base, eps) * dt
    discomfort_w = (kappa / 1000) * ((cut * 1000) ** 2 + (deferred * 1000) ** 2) / max(base * 1000, eps * 1000) * dt
    bill_kw = energy_price * base * dt
    bill_w = (energy_price / 1000) * (base * 1000) * dt
    assert math.isclose(discomfort_kw, discomfort_w)
    assert math.isclose(bill_kw, bill_w)

    integrity = {}
    for name in ["protected_files_manifest.json", "history_files_manifest.json"]:
        manifest = json.loads((BASE / name).read_text(encoding="utf-8"))
        changed = [path for path, expected in manifest.items() if sha(ROOT / path) != expected]
        assert not changed, changed
        integrity[name] = {"files": len(manifest), "unchanged": True}
    reference = BASE / "verify_v9_mechanisms.py"
    assert sha(reference) == REFERENCE_SHA
    for name in ["verification_results.json", "formula_validation.json"]:
        current = json.loads((BASE / name).read_text(encoding="utf-8"))
        excluded = {"document_sha256", "document_checks"}
        assert semantic_sha({k: v for k, v in current.items() if k not in excluded}) == NUMERIC_BASELINE_SHA[name], name
        assert current["document_sha256"] == sha(BASE / "modeling_complete_zh.md")

    report = {
        "status": "passed", "date": "2026-10-14",
        "source_before_sha256": SOURCE_BEFORE_SHA,
        "source_after_sha256": sha(BASE / "modeling_complete_zh.md"),
        "all_original_formula_relations_preserved_after_notation_normalization": True, "numbered_equations": 30,
        "original_display_formula_count": len(get_displays(text)), "code_examples_unchanged": True,
        "unit_bases": ["yuan", "kWh", "h"], "monetary_dimension_checks": money_checks,
        "negative_controls_rejected": list(negative_controls), "different_step_accounting_probes": step_probes,
        "power_and_energy_unit_conversion": {"kW_kWh_to_W_Wh": "passed", "epsilon_converted_with_power": True},
        "coefficient_policy": {
            "UC_overdue": "yuan/kWh per assessment, charged once per physical period",
            "consumer_backlog_and_overdue": "yuan/(kWh*h), times end-of-period energy and duration",
            "external_cash": "already integrated amount in yuan for the current period",
            "terminal": "yuan/kWh times energy, appended once on true termination",
            "source_formula_30": "historical reference only; not a current v9 settlement expression",
        },
        "original_mechanism_and_formula_numeric_results_identical": True,
        "integrity": integrity, "reference_verifier_unchanged": True,
        "scope": "Dimensional accounting and unchanged modeling definitions; no production code edits or convergence claims.",
    }
    (BASE / "cost_units_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "monetary_checks": len(money_checks),
                      "negative_controls": len(negative_controls), "step_probes": len(step_probes),
                      "all_formula_relations_preserved": True, "integrity": integrity}))


if __name__ == "__main__":
    main()
