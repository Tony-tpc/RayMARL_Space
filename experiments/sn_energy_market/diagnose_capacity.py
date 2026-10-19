"""Compare measured load with physical capacity, without training or test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "environments/sn_energy_market/src"))
sys.path.insert(0, str(ROOT / "RLAlgo/src"))

from experiments.sn_energy_market.profile_dataset import ProfileDataset


def oracle(load, pv, system, *, retain_initial=False):
    """Minimize lost energy with perfect foresight and aggregate physical dispatch.

    Variables are grid import, PV use, charge, discharge, lost power and stored
    energy. Mode exclusivity is relaxed; the result records simultaneous power
    so callers can see whether the relaxation changes physical feasibility.
    """
    n = len(load)
    dt = system.resolution_minutes / 60
    grid, used, charge, discharge, lost, energy = [i * n for i in range(6)]
    objective = np.zeros(6 * n + 1)
    objective[lost : lost + n] = dt
    matrix = lil_matrix((2 * n + 1, len(objective)))
    rhs = np.zeros(2 * n + 1)
    for t in range(n):
        matrix[t, grid + t] = 1
        matrix[t, used + t] = 1
        matrix[t, discharge + t] = 1
        matrix[t, lost + t] = 1
        matrix[t, charge + t] = -1
        rhs[t] = load[t]
        matrix[n + t, energy + t + 1] = 1
        matrix[n + t, energy + t] = -1
        matrix[n + t, charge + t] = -dt * system.ess_charge_efficiency
        matrix[n + t, discharge + t] = dt / system.ess_discharge_efficiency
    matrix[2 * n, energy] = 1
    rhs[2 * n] = system.ess_initial_energy_kwh
    bounds = (
        [(0, system.grid_import_max_kw)] * n
        + [(0, float(x)) for x in pv]
        + [(0, system.ess_charge_max_kw)] * n
        + [(0, system.ess_discharge_max_kw)] * n
        + [(0, float(x)) for x in load]
        + [(system.ess_energy_min_kwh, system.ess_energy_capacity_kwh)] * (n + 1)
    )
    if retain_initial:
        bounds[-1] = (system.ess_initial_energy_kwh, system.ess_energy_capacity_kwh)
    result = linprog(
        objective, A_eq=matrix.tocsr(), b_eq=rhs, bounds=bounds, method="highs"
    )
    if not result.success:
        raise RuntimeError(result.message)
    simultaneous = np.minimum(
        result.x[charge : charge + n], result.x[discharge : discharge + n]
    ).max()
    return {
        "minimum_unserved_energy_kwh": float(result.fun),
        "maximum_service_rate": float(1 - result.fun / (load.sum() * dt)),
        "initial_soc_kwh": system.ess_initial_energy_kwh,
        "final_soc_kwh": float(result.x[-1]),
        "require_final_soc_at_least_initial": retain_initial,
        "simultaneous_charge_discharge_max_kw": float(simultaneous),
        "balance_residual_max": float(abs(matrix @ result.x - rhs).max()),
    }


def stats(load, pv, grid, dt):
    deficit = np.maximum(load - grid - pv, 0)
    return {
        "periods": len(load),
        "load_energy_kwh": float(load.sum() * dt),
        "pv_energy_kwh": float(pv.sum() * dt),
        "load_peak_kw": float(load.max()),
        "load_q95_kw": float(np.quantile(load, 0.95)),
        "grid_only_insufficient_periods": int((load > grid + 1e-10).sum()),
        "grid_pv_insufficient_periods": int((deficit > 1e-10).sum()),
        "grid_pv_deficit_energy_kwh": float(deficit.sum() * dt),
        "grid_pv_deficit_peak_kw": float(deficit.max()),
        "no_storage_max_service_rate": float(1 - deficit.sum() / load.sum()),
        "pv_at_load_peak_kw": float(pv[load.argmax()]),
    }


def greedy(load, pv, system):
    """Charge on instantaneous surplus and discharge on deficit; no future data."""
    dt = system.resolution_minutes / 60
    energy = system.ess_initial_energy_kwh
    lost = 0.0
    lost_periods = 0
    for demand, solar in zip(load, pv):
        spare = system.grid_import_max_kw + solar - demand
        if spare >= 0:
            charge = min(
                spare,
                system.ess_charge_max_kw,
                (system.ess_energy_capacity_kwh - energy)
                / (dt * system.ess_charge_efficiency),
            )
            energy += charge * dt * system.ess_charge_efficiency
        else:
            discharge = min(
                -spare,
                system.ess_discharge_max_kw,
                (energy - system.ess_energy_min_kwh)
                * system.ess_discharge_efficiency
                / dt,
            )
            energy -= discharge * dt / system.ess_discharge_efficiency
            residual = -spare - discharge
            lost += residual * dt
            lost_periods += residual > 1e-10
    return {
        "unserved_energy_kwh": float(lost),
        "service_rate": float(1 - lost / (load.sum() * dt)),
        "unserved_periods": int(lost_periods),
        "final_soc_kwh": float(energy),
        "future_data_used": False,
        "market_allocation_included": False,
    }


def diagnose(dataset):
    system = dataset.environment_config().system
    dt = system.resolution_minutes / 60
    annual, windows = {}, {}
    annual_periods = window_periods = lp_solves = lp_periods = 0
    for split in ("train", "validation"):
        arrays = dataset._arrays[split]
        load, pv = arrays["loads_kw"].sum(axis=1), arrays["der_kw"]
        annual[split] = {
            "base": stats(load, pv, system.grid_import_max_kw, dt),
            "rigid": stats(0.8 * load, pv, system.grid_import_max_kw, dt),
            "base_greedy_physical": greedy(load, pv, system),
        }
        annual_periods += len(load)
    for offset in (0, 3, 6, 9):
        window = dataset.windows("validation")[offset]
        arrays = dataset._arrays["validation"]
        selection = slice(window.start, window.start + 96)
        load = arrays["loads_kw"][selection].sum(axis=1)
        pv = arrays["der_kw"][selection]
        row = {
            "start": window.start_datetime,
            "seed": 10042 + offset,
            "base": stats(load, pv, system.grid_import_max_kw, dt),
            "rigid": stats(0.8 * load, pv, system.grid_import_max_kw, dt),
            "base_greedy_physical": greedy(load, pv, system),
        }
        for label, target, preserve_soc in (
            ("base_oracle", load, False),
            ("base_oracle_soc_preserved", load, True),
            ("rigid_oracle", 0.8 * load, False),
        ):
            row[label] = oracle(target, pv, system, retain_initial=preserve_soc)
            lp_solves += 1
            lp_periods += len(load)
        windows[window.name] = row
        window_periods += len(load)
    return {
        "loaded_splits": list(dataset.splits),
        "test_loaded": False,
        "training_started": False,
        "method": (
            "Train/validation meter profiles only; full-base and rigid (.8 base) "
            "capacity comparisons. LP is a perfect-foresight aggregate physical "
            "benchmark, not a policy or market outcome; no shifted backlog. "
            "Initial battery is half full. Mode exclusivity is relaxed and "
            "simultaneous charge/discharge is reported. Greedy dispatch uses "
            "only current resource surplus or deficit. Annual greedy runs "
            "continuously within each year; four pilot windows each reset SOC."
        ),
        "counts": {
            "unique_loaded_profile_periods": annual_periods,
            "validation_window_periods": window_periods,
            "linear_program_solves": lp_solves,
            "linear_program_period_instances": lp_periods,
            "greedy_dispatch_period_instances": annual_periods + window_periods,
            "production_environment_steps": 0,
            "rl_optimizer_updates": 0,
        },
        "system": {
            "grid_max_kw": system.grid_import_max_kw,
            "ess_power_kw": system.ess_discharge_max_kw,
            "ess_capacity_kwh": system.ess_energy_capacity_kwh,
            "ess_initial_kwh": system.ess_initial_energy_kwh,
            "charge_efficiency": system.ess_charge_efficiency,
            "discharge_efficiency": system.ess_discharge_efficiency,
        },
        "data_sha256": {
            "manifest": dataset.manifest_sha256,
            **{
                split: dataset.manifest["splits"][split]["sha256"]
                for split in dataset.splits
            },
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("profile_dataset.py"))
        },
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "annual": annual,
        "windows": windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets/ausgrid/processed/manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/ausgrid-diagnosis/capacity.json",
    )
    args = parser.parse_args()
    dataset = ProfileDataset(args.dataset, splits=("train", "validation"))
    report = diagnose(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
