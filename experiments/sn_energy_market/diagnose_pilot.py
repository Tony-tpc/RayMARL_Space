"""Frozen Ausgrid pilot diagnosis; no optimizer, training or test split access."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "RLAlgo/src"),
    str(ROOT / "environments/sn_energy_market/src"),
]

from raymarl.algorithms.stackelberg.inference import ActorBundle
from snmarket import EnergyMarketEngine
from snmarket.contracts import Ledger

from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)

PILOT = ROOT / "outputs/ausgrid-pilot/20261019-verified"
OUTPUT = ROOT / "outputs/ausgrid-diagnosis"
SEEDS = (10042, 10045, 10048, 10051)
GAMMA = 0.9801


def summarize(record, config):
    dt = config.system.dt_hours
    values = {
        f"ledger/{f.name}": getattr(record.ledger, f.name)
        for f in fields(Ledger)
        if f.name != "consumers"
    }
    values.update(
        {
            "native_kwh": sum(o.native_kw for o in record.orders) * dt,
            "lost_kwh": sum(s.native_lost_kw for s in record.services.values()) * dt,
            "repay_kwh": sum(s.repay_actual_kw for s in record.services.values()) * dt,
            "overdue_kwh_periods": sum(
                s.overdue_after_kwh for s in record.services.values()
            ),
            "der_sales": sum(c.der_payment for c in record.ledger.consumers.values()),
            "uc_sales": sum(c.uc_payment for c in record.ledger.consumers.values()),
            "consumer_cost": sum(
                c.total_cost for c in record.ledger.consumers.values()
            ),
            "pg_offer_kw_periods": record.uc_action.grid_import_offer_kw,
            "periods": 1,
        }
    )
    for field in (
        "grid_buy_kw",
        "der_generation_kw",
        "der_planned_curtailment_kw",
        "der_unsold_kw",
        "der_spill_kw",
        "der_to_ess_kw",
        "grid_to_ess_kw",
        "ess_discharge_kw",
        "supply_served_kw",
    ):
        values[field.replace("_kw", "_kwh")] = getattr(record.physical, field) * dt
    if values["lost_kwh"] > 1e-8:
        values["lost_periods"] = 1
        values["charge_while_lost_kwh"] = record.uc_action.charge_kw * dt
        values["active_curtail_while_lost_kwh"] = (
            record.uc_action.der_planned_curtailment_kw * dt
        )
    assert abs(record.physical.power_residual_kw) < 1e-8
    assert abs(record.ledger.cash_residual) < 1e-8
    return values


def add(total, row):
    for key, value in row.items():
        total[key] += float(value)


def finish(total):
    return {**dict(total), "service_rate": 1 - total["lost_kwh"] / total["native_kwh"]}


def service_controls(observation, config):
    """Public aggregate observation only; no future data or private household state.

    Announce all PG. Charge only PV beyond baseline plus due repayment. Discharge
    for any remaining aggregate deficit. DR procurement is zero in this baseline.
    """
    s = config.system
    energy, generation, grid = map(float, observation[2:5])
    demand = float(observation[8] + observation[11])
    charge = min(
        max(0.0, generation - demand),
        s.ess_charge_max_kw,
        max(0.0, s.ess_energy_capacity_kwh - energy)
        / (s.ess_charge_efficiency * s.dt_hours),
    )
    discharge_capacity = min(
        s.ess_discharge_max_kw,
        s.ess_discharge_efficiency
        * max(0.0, energy - s.ess_energy_min_kwh)
        / s.dt_hours,
    )
    if charge > 1e-8:
        return np.array([1, 0, charge / generation, 0, 1, 0])
    deficit = max(0.0, demand - generation - grid)
    signed = -min(1.0, deficit / discharge_capacity) if discharge_capacity > 0 else 0
    return np.array([1, 0, 0, 0, signed, 0])


def counterfactuals(engine, snapshot, record):
    a = record.uc_action
    pg = engine.config.system.grid_import_max_kw
    alternatives = {
        "grid_full": replace(a, grid_import_offer_kw=pg),
        "curtail_zero": replace(a, der_planned_curtailment_kw=0),
        "charge_zero": replace(a, der_to_ess_kw=0, grid_to_ess_kw=0),
        "combined": replace(
            a,
            grid_import_offer_kw=pg,
            der_planned_curtailment_kw=0,
            der_to_ess_kw=0,
            grid_to_ess_kw=0,
        ),
        "dr_zero": replace(a, dr_procurement_cap_kw=0),
    }
    clone = EnergyMarketEngine(engine.config, engine.profiles)
    for name, action in alternatives.items():
        clone.set_state(snapshot)
        clone.apply_uc_action(action)
        result = clone.apply_consumer_actions(record.consumer_actions)
        yield name, summarize(result, engine.config)


def rollout(bundle, factory, seed, variant, expected):
    env = factory(seed)
    observation = env.reset(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    totals, future = defaultdict(float), defaultdict(float)
    alternatives = defaultdict(lambda: defaultdict(float))
    action_rows = []
    # Extend only the original frozen trajectory for same-state critic diagnosis.
    length = 192 if variant == "sampled" else 96
    for period in range(length):
        snapshot = (
            env.engine.get_state() if variant == "sampled" and period < 96 else None
        )
        deterministic = variant == "mean"
        # Retain UC draw call ordering. Changed Beta parameters can still consume
        # different random draws; only the one-step cases fix Consumer actions.
        raw = bundle.uc_controls(observation, deterministic=deterministic)
        if variant == "full_grid":
            raw = np.array([1, 0, 0, 0, 0, 0])
        elif variant in ("service_rule", "passive_consumers"):
            raw = service_controls(observation, env.config)
        decision = env.publish(raw)
        actions = {
            key: bundle.consumer_action(
                key,
                decision.consumer_observations[key],
                decision.consumer_bounds[key],
                deterministic=deterministic,
            )
            for key in env.consumer_ids
        }
        if variant == "passive_consumers":
            actions = {
                key: np.array([0, 0, 1, decision.consumer_bounds[key][10]])
                for key in env.consumer_ids
            }
        result = env.step(actions)
        record = env.engine.latest_record
        row = summarize(record, env.config)
        target = totals if period < 96 else future
        add(target, row)
        target["discounted_uc_return"] += GAMMA ** (period % 96) * result.rewards["uc"]
        if period < 96:
            action_rows.append(decision.uc_action.tolist())
        if snapshot is not None:
            for name, alternative in counterfactuals(env.engine, snapshot, record):
                add(alternatives[name], alternative)
                alternatives[name]["reward_improved_periods"] += (
                    alternative["ledger/uc_reward"] > row["ledger/uc_reward"] + 1e-8
                )
                alternatives[name]["reward_decreased_periods"] += (
                    alternative["ledger/uc_reward"] < row["ledger/uc_reward"] - 1e-8
                )
        observation = result.next_uc_observation
    if variant == "sampled":
        np.testing.assert_allclose(
            [
                totals["ledger/cash_profit"],
                totals["lost_kwh"],
                totals["discounted_uc_return"],
            ],
            [
                expected["cash_profit"],
                expected["native_lost_kwh"],
                expected["returns"]["uc"],
            ],
            rtol=1e-9,
            atol=1e-8,
        )
    return {
        "seed": seed,
        "variant": variant,
        "totals": finish(totals),
        "mean_physical_action": np.mean(action_rows, axis=0).tolist(),
        "one_step_counterfactuals": {
            key: finish(value) for key, value in alternatives.items()
        },
        "critic_diagnosis": {
            "prediction_at_period_96": expected["bootstrap_values"]["uc"],
            "same_state_following_96_step_sampled_return": future[
                "discounted_uc_return"
            ],
            "note": "Single frozen rollout, truncated MC sample, not the exact expected value.",
        }
        if variant == "sampled"
        else None,
    }


def main():
    torch.set_num_threads(1)
    dataset = ProfileDataset(
        ROOT / "datasets/ausgrid/processed/manifest.json",
        splits=("train", "validation"),
    )
    factory = DatasetEnvFactory(dataset, seed=42)
    bundle = ActorBundle(PILOT / "actors")
    weights_before = {
        key: {
            name: value.detach().clone() for name, value in module.state_dict().items()
        }
        for key, module in bundle.modules.items()
    }
    original = json.loads((PILOT / "report.json").read_text(encoding="utf-8"))
    expected = {row["seed"]: row for row in original["validation_after"]["rows"]}
    rows = []
    for variant in (
        "sampled",
        "mean",
        "full_grid",
        "service_rule",
        "passive_consumers",
    ):
        for seed in SEEDS:
            row = rollout(bundle, factory, seed, variant, expected[seed])
            rows.append(row)
            print(
                variant,
                seed,
                round(row["totals"]["service_rate"], 6),
                round(row["totals"]["ledger/cash_profit"], 6),
                flush=True,
            )
    for key, module in bundle.modules.items():
        assert all(
            torch.equal(value, weights_before[key][name])
            for name, value in module.state_dict().items()
        )
    aggregates = {}
    for variant in {row["variant"] for row in rows}:
        total = defaultdict(float)
        for row in rows:
            if row["variant"] == variant:
                add(
                    total,
                    {
                        key: value / 4
                        for key, value in row["totals"].items()
                        if key != "service_rate"
                    },
                )
        aggregates[variant] = finish(total)
    counter = {}
    for name in rows[0]["one_step_counterfactuals"]:
        total = defaultdict(float)
        for row in rows:
            if row["variant"] == "sampled":
                add(
                    total,
                    {
                        key: value / 4
                        for key, value in row["one_step_counterfactuals"][name].items()
                        if key != "service_rate"
                    },
                )
        counter[name] = finish(total)
    result = {
        "loaded_splits": dataset.splits,
        "optimizer_steps": 0,
        "actor_parameters_unchanged": True,
        "pilot_replay_matched": True,
        "validation_trajectory_periods": 2304,
        "one_step_counterfactual_periods": 1920,
        "averages_per_48h_window": aggregates,
        "one_step_counterfactual_averages": counter,
        "rows": rows,
        "manifest_sha256": dataset.manifest_sha256,
        "actor_sha256": hashlib.sha256(
            (PILOT / "actors/actors.pt").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes": [
            "All production parameters unchanged.",
            "One-step counterfactuals retain the original state and Consumer physical actions.",
            "Whole-trajectory rules retain frozen Consumer policies, which respond to new observations.",
            "Rule actions are explicit diagnostic policies, never automatic environment overrides.",
            "Rule comparisons are not optimized policies or equilibrium evidence.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "policy_diagnosis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
