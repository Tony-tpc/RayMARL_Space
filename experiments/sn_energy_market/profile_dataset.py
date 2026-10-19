"""Measured profile datasets and explicit experiment assumptions, independent of RL."""

from __future__ import annotations

import calendar
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
from snmarket import EnvironmentConfig, Profiles
from snmarket.contracts import ConsumerAction

from experiments.sn_energy_market.training_env import MarketTrainingEnv


@dataclass(frozen=True)
class MonthWindow:
    split: str
    name: str
    start: int
    stop: int
    start_datetime: str

    @property
    def periods(self):
        return self.stop - self.start


class ProfileDataset:
    """Read fixed-identity, chronological splits and expose complete calendar months.

    NPZ files contain measured ``loads_kw``, ``der_kw`` and interval-start
    ``timestamps``. Flexible capacity and procurement prices are separate scenario
    assumptions. Resource calibration uses only the training split. Only selected
    split files are opened; the complete manifest remains part of the data contract.
    """

    def __init__(self, manifest_path, *, splits=("train", "validation", "test")):
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.manifest_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.name = self.manifest["dataset"]
        self.time_policy = self.manifest["time_policy"]
        split_order = ("train", "validation", "test")
        self.splits = tuple(split for split in split_order if split in splits)
        if not self.splits or len(self.splits) != len(splits):
            raise ValueError("Select distinct train, validation or test splits")
        if any(
            self.manifest["units"][field] != "kW" for field in ("loads_kw", "der_kw")
        ):
            raise ValueError("Dataset loads_kw and der_kw must contain power in kW")
        self.household_ids = tuple(self.manifest["household_ids"])
        self.resolution_minutes = int(self.manifest["resolution_minutes"])
        if self.resolution_minutes < 1 or 1440 % self.resolution_minutes:
            raise ValueError("Dataset resolution must be a positive divisor of a day")
        if not self.household_ids or len(set(self.household_ids)) != len(
            self.household_ids
        ):
            raise ValueError("Dataset household identities must be nonempty and unique")
        self._arrays, self._windows = {}, {}
        ranges = [
            tuple(
                datetime.fromisoformat(self.manifest["splits"][split][field])
                for field in ("start_datetime", "end_datetime")
            )
            for split in split_order
        ]
        if any(left[1] > right[0] for left, right in pairwise(ranges)):
            raise ValueError(
                "Training, validation and test dates must be ordered and disjoint"
            )
        for split in self.splits:
            entry = self.manifest["splits"][split]
            path = self.path.parent / entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Dataset checksum differs: {split}")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    key: archive[key] for key in ("loads_kw", "der_kw", "timestamps")
                }
            loads, der = arrays["loads_kw"], arrays["der_kw"]
            if (
                loads.shape != (entry["periods"], len(self.household_ids))
                or not len(loads)
                or der.shape != (len(loads),)
                or arrays["timestamps"].shape != (len(loads),)
                or not np.isfinite(loads).all()
                or not np.isfinite(der).all()
                or (loads < 0).any()
                or (der < 0).any()
            ):
                raise ValueError(
                    f"Dataset profile dimensions or values differ: {split}"
                )
            start = datetime.fromisoformat(entry["start_datetime"])
            step = timedelta(minutes=self.resolution_minutes)
            if start.tzinfo is None or start.second or start.microsecond:
                raise ValueError("Dataset requires fixed-offset, day-aligned intervals")
            expected = [start + i * step for i in range(len(loads))]
            actual = [
                datetime.fromisoformat(str(item)) for item in arrays["timestamps"]
            ]
            if any(
                a != b or a.utcoffset() != start.utcoffset()
                for a, b in zip(actual, expected)
            ):
                raise ValueError(
                    f"Dataset timestamps are not contiguous on the declared clock: {split}"
                )
            end = start + len(loads) * step
            if datetime.fromisoformat(entry["end_datetime"]) != end:
                raise ValueError(
                    f"Dataset end_datetime must be the exclusive boundary: {split}"
                )
            windows = []
            offset, cursor = 0, start
            while cursor < end:
                length = calendar.monthrange(cursor.year, cursor.month)[1] * (
                    1440 // self.resolution_minutes
                )
                if (
                    cursor.day != 1
                    or cursor.hour
                    or cursor.minute
                    or offset + length > len(loads)
                ):
                    raise ValueError(
                        f"Dataset splits must contain complete calendar months: {split}"
                    )
                windows.append(
                    MonthWindow(
                        split,
                        cursor.strftime("%Y-%m"),
                        offset,
                        offset + length,
                        cursor.isoformat(),
                    )
                )
                cursor += length * step
                offset += length
            self._windows[split] = tuple(windows)
            for array in arrays.values():
                array.flags.writeable = False
            self._arrays[split] = arrays

    def windows(self, split):
        return self._windows[split]

    def calibration(self, flex_ratio=0.2):
        """A declared resource scenario fitted only to training gross consumption."""
        loads = self._arrays["train"]["loads_kw"]
        peak = float(np.quantile(loads.sum(axis=1), 0.95, method="linear"))
        return {
            "fit_split": "train",
            "aggregate_load_q95_kw": peak,
            "quantile_method": "linear",
            "flex_ratio": flex_ratio,
            "grid_import_max_kw": peak,
            "ess_power_kw": 0.25 * peak,
            "ess_energy_capacity_kwh": 0.5 * peak,
            "dr_procurement_max_kw": flex_ratio * peak,
            "shift_backlog_limit_kwh": (
                24 * flex_ratio * np.quantile(loads, 0.95, axis=0, method="linear")
            ).tolist(),
            "resource_ratios": {
                "grid_to_q95_load": 1.0,
                "ess_power_to_q95_load": 0.25,
                "ess_duration_hours": 2.0,
                "initial_soc_fraction": 0.5,
                "queue_flexible_hours": 24.0,
            },
        }

    def environment_config(self, base=None):
        """Convert cadence; calibrate defaults or retain explicitly supplied resources."""
        calibrate_resources = base is None
        base = base or EnvironmentConfig()
        scale = self.resolution_minutes / base.system.resolution_minutes
        calibration = self.calibration(base.system.dr_limit_ratio)
        window = self.windows("train")[0]
        resources = (
            {
                "grid_import_max_kw": calibration["grid_import_max_kw"],
                "ess_charge_max_kw": calibration["ess_power_kw"],
                "ess_discharge_max_kw": calibration["ess_power_kw"],
                "ess_energy_capacity_kwh": calibration["ess_energy_capacity_kwh"],
                "ess_energy_min_kwh": 0.0,
                "ess_initial_energy_kwh": calibration["ess_energy_capacity_kwh"] / 2,
            }
            if calibrate_resources
            else {}
        )
        system = replace(
            base.system,
            resolution_minutes=self.resolution_minutes,
            num_consumers=len(self.household_ids),
            horizon_steps=window.periods,
            episode_mode="finite",
            start_datetime=window.start_datetime,
            **resources,
        )
        boundaries = [
            base.market.shift_deadline_steps / scale,
            *(
                value / scale
                for pair in base.market.dr_event_intervals
                for value in pair
            ),
        ]
        if any(not value.is_integer() for value in boundaries):
            raise ValueError(
                "Deadline and event boundaries must align to dataset resolution"
            )
        market = replace(
            base.market,
            shift_deadline_steps=int(boundaries[0]),
            dr_event_intervals=tuple(
                tuple(int(value / scale) for value in pair)
                for pair in base.market.dr_event_intervals
            ),
            dr_procurement_max_kw=calibration["dr_procurement_max_kw"]
            if calibrate_resources
            else base.market.dr_procurement_max_kw,
            shift_backlog_limit_kwh=tuple(calibration["shift_backlog_limit_kwh"])
            if calibrate_resources
            else base.market.shift_backlog_limit_kwh,
        )
        rewards = replace(
            base.rewards,
            uc_due_rate_per_kwh=base.rewards.uc_due_rate_per_kwh * scale,
        )
        return replace(base, system=system, market=market, rewards=rewards).validate()

    def profiles(self, window, config):
        arrays = self._arrays[window.split]
        loads = arrays["loads_kw"][window.start : window.stop]
        flex = loads * config.system.dr_limit_ratio
        start = datetime.fromisoformat(window.start_datetime)
        hours = (
            np.arange(window.periods) * config.system.dt_hours
            + start.hour
            + start.minute / 60
        ) % 24
        prices = np.where((hours >= 17) & (hours < 22), 0.65, 0.35)
        return Profiles(
            loads,
            arrays["der_kw"][window.start : window.stop],
            prices,
            flex,
            rigid_kw=loads - flex,
            start_datetime=window.start_datetime,
            resolution_minutes=self.resolution_minutes,
            metadata={
                "source": self.name,
                "value_kind": "realized",
                "split": window.split,
                "window": window.name,
                "household_ids": self.household_ids,
                "manifest_sha256": self.manifest_sha256,
                "profile_file_sha256": self.manifest["splits"][window.split]["sha256"],
                "flexibility": "scenario_fraction_of_gross_consumption",
                "grid_price": "scenario_0.35_offpeak_0.65_17_to_22_AUD_per_kWh",
                "time_policy": self.time_policy,
            },
        )


class DatasetEnvFactory:
    """Bind scenario seeds to complete months and optional training start dates.

    Positive start days count from the beginning of a month; -1 means its last day,
    -2 its penultimate day. A fixed, nonlearning rule runs from month start to each
    selected date. Its actual state/history is the reset state, and termination
    remains at month end. Validation and test scenarios always start on day one.
    """

    def __init__(self, dataset, config=None, *, seed=42, training_start_days=(1,)):
        self.dataset = dataset
        self.config = config or dataset.environment_config()
        self.training_start_days = tuple(training_start_days)
        if (
            not self.training_start_days
            or len(set(self.training_start_days)) != len(self.training_start_days)
            or any(type(day) is not int or day == 0 for day in self.training_start_days)
        ):
            raise ValueError("Training start days must be distinct nonzero integers")
        if (
            self.config.system.num_consumers != len(dataset.household_ids)
            or self.config.system.resolution_minutes != dataset.resolution_minutes
        ):
            raise ValueError(
                "Experiment configuration must match dataset identities and resolution"
            )
        self.seed_splits, self._by_seed, self._start_periods = {}, {}, {}
        self._initial_snapshots = {}
        for index, split in enumerate(("train", "validation", "test")):
            if split not in dataset.splits:
                continue
            windows = dataset.windows(split)
            start_days = self.training_start_days if split == "train" else (1,)
            scenarios = []
            for window in windows:
                days = window.periods // self.config.system.steps_per_day
                for requested_day in start_days:
                    day = requested_day if requested_day > 0 else days + requested_day + 1
                    if not 1 <= day <= days:
                        raise ValueError(f"Training start day {requested_day} is outside {window.name}")
                    scenarios.append((window, (day - 1) * self.config.system.steps_per_day))
            seeds = tuple(
                range(seed + index * 10000, seed + index * 10000 + len(scenarios))
            )
            self.seed_splits[split] = seeds
            self._by_seed.update((key, value[0]) for key, value in zip(seeds, scenarios))
            self._start_periods.update((key, value[1]) for key, value in zip(seeds, scenarios))
            if split == "train":
                # Traverse the year first, rotating the within-month start stratum.
                width = len(start_days)
                self.seed_splits[split] = tuple(
                    seeds[month * width + (month + cycle) % width]
                    for cycle in range(width)
                    for month in range(len(windows))
                )

    def __getstate__(self):
        # A cache speeds repeated scenario resets, but is not experiment state.
        return {**vars(self), "_initial_snapshots": {}}

    def profiles_for_seed(self, seed):
        """Read scenario data without constructing a training reset state."""
        return self.dataset.profiles(self._by_seed[seed], self.config)

    def __call__(self, seed):
        window = self._by_seed[seed]
        config = replace(
            self.config,
            system=replace(
                self.config.system,
                horizon_steps=window.periods,
                start_datetime=window.start_datetime,
            ),
        )
        profiles = self.profiles_for_seed(seed)
        start_period = self._start_periods[seed]
        if start_period and seed not in self._initial_snapshots:
            env = MarketTrainingEnv(config, profiles)
            env.reset(seed)
            action = ConsumerAction(
                curtail_kw=0.0,
                signed_shift_kw=0.0,
                der_preference_fraction=1.0,
                der_bid_per_kwh=config.market.der_reference_rate_per_kwh,
            )
            for _ in range(start_period):
                env.publish([1, 0, 0, 0, 0, 0])
                env.step(dict.fromkeys(env.consumer_ids, action))
            self._initial_snapshots[seed] = env.get_state()
        return MarketTrainingEnv(
            config, profiles, initial_snapshot=self._initial_snapshots.get(seed)
        )

    def get_contract(self):
        """Portable data, identity, calendar and assumption hashes for checkpoints."""
        contract = {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "manifest_sha256": self.dataset.manifest_sha256,
            "loaded_splits": self.dataset.splits,
            "household_ids": self.dataset.household_ids,
            "config": self.config.to_dict(),
            "resource_calibration": {
                "training_reference": self.dataset.calibration(
                    self.config.system.dr_limit_ratio
                ),
                "applied": {
                    "grid_import_max_kw": self.config.system.grid_import_max_kw,
                    "ess_charge_max_kw": self.config.system.ess_charge_max_kw,
                    "ess_discharge_max_kw": self.config.system.ess_discharge_max_kw,
                    "ess_energy_capacity_kwh": self.config.system.ess_energy_capacity_kwh,
                    "ess_energy_min_kwh": self.config.system.ess_energy_min_kwh,
                    "ess_initial_energy_kwh": self.config.system.ess_initial_energy_kwh,
                    "dr_procurement_max_kw": self.config.market.dr_procurement_max_kw,
                    "shift_backlog_limit_kwh": self.config.market.shift_backlog_limit_kwh,
                },
            },
            "scenario_prices": {
                "currency": "AUD",
                "offpeak_per_kwh": 0.35,
                "peak_per_kwh": 0.65,
                "peak_hours": [17, 22],
                "all_monetary_coefficients": "AUD scenario values, not observed tariffs",
            },
            "cadence_assumption": "UC overdue assessments preserve hourly strength; energy rates, hourly rates and already specified per-period amounts unchanged",
            "seed_windows": {
                seed: {
                    **vars(window),
                    "start_period": self._start_periods[seed],
                    "start_day": self._start_periods[seed] // self.config.system.steps_per_day + 1,
                    "sampling_start_datetime": (
                        datetime.fromisoformat(window.start_datetime)
                        + timedelta(minutes=self._start_periods[seed] * self.dataset.resolution_minutes)
                    ).isoformat(),
                }
                for seed, window in self._by_seed.items()
            },
            "training_start_days": self.training_start_days,
            "initial_state_rule": {
                "uc_controls": [1, 0, 0, 0, 0, 0],
                "consumer": "zero cut/shift; DER fraction one; DER floor bid",
                "state": "actual monthly UC energy and history; empty queues; configured initial ESS",
                "warmup": "fixed-rule state construction, excluded from learning samples",
            },
            "source_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in ("profile_dataset.py", "training_env.py")
            },
            "episode_boundary": "complete month; optional fixed-rule warm start; terminal valuation once at month end",
        }
        return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))
