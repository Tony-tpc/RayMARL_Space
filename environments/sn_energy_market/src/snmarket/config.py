"""v9 physical/economic configuration; individual coefficients can be scalar or tuple."""
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from numbers import Real
from pathlib import Path

VERSION = "v9_announced_supply_adequacy"
SCHEMA_VERSION = 9

def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")

def individual(value, index):
    return float(value[index] if isinstance(value, (tuple, list)) else value)

def immutable_value(value):
    return tuple(immutable_value(x) for x in value) if isinstance(value, (tuple, list)) else value

def personal(value, name, count):
    if isinstance(value, (tuple, list)):
        if len(value) != count:
            raise ValueError(f"{name} must contain {count} values")
        for entry in value:
            nonnegative(entry, name)
    else:
        nonnegative(value, name)

@dataclass(frozen=True)
class SystemConfig:
    resolution_minutes: int = 15
    num_consumers: int = 8
    horizon_steps: int = 96
    episode_mode: str = "finite"
    grid_import_max_kw: float = 100.0
    ess_energy_capacity_kwh: float = 200.0
    ess_energy_min_kwh: float = 0.0
    ess_initial_energy_kwh: float = 100.0
    ess_charge_max_kw: float = 25.0
    ess_discharge_max_kw: float = 25.0
    ess_charge_efficiency: float = 0.95
    ess_discharge_efficiency: float = 0.95
    dr_limit_ratio: float = 0.20
    start_datetime: str = "2026-01-01T00:00:00+08:00"

    @property
    def dt_hours(self):
        return self.resolution_minutes / 60

    @property
    def steps_per_day(self):
        return 1440 // self.resolution_minutes

    def validate(self):
        for name in ("resolution_minutes", "num_consumers", "horizon_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if 1440 % self.resolution_minutes:
            raise ValueError("resolution_minutes must divide a day")
        if self.episode_mode not in {"finite", "continuous"}:
            raise ValueError("episode_mode must be finite or continuous")
        for f in fields(self):
            if f.name not in {"episode_mode", "start_datetime"}:
                nonnegative(getattr(self, f.name), f.name)
        if not 0 <= self.ess_energy_min_kwh <= self.ess_initial_energy_kwh <= self.ess_energy_capacity_kwh:
            raise ValueError("Inconsistent ESS initial energy/bounds")
        for name in ("ess_charge_efficiency", "ess_discharge_efficiency"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if not 0 <= self.dr_limit_ratio <= 1:
            raise ValueError("dr_limit_ratio must be in [0,1]")
        start = datetime.fromisoformat(self.start_datetime)
        if (start.tzinfo is None or start.second or start.microsecond
                or (start.hour * 60 + start.minute) % self.resolution_minutes):
            raise ValueError("start_datetime requires a timezone and an aligned interval boundary")

@dataclass(frozen=True)
class MarketConfig:
    mechanism_version: str = VERSION
    der_bid_cap: float | tuple[float, ...] = 1.50
    dr_procurement_max_kw: float = 40.0
    shift_backlog_limit_kwh: float | tuple[float, ...] = 25.0
    shift_deadline_steps: int = 96
    dr_event_intervals: tuple[tuple[int, int], ...] = ((68, 80),)
    tier_thresholds_kwh: tuple[float, ...] = (100.0, 200.0)
    uc_flat_rates: tuple[float, ...] = (0.40, 0.50, 0.60)
    uc_high_rates: tuple[float, ...] = (0.60, 0.70, 0.80)
    uc_peak_rates: tuple[float, ...] = (0.70, 0.80, 0.90)
    uc_high_intervals_hours: tuple[tuple[float, float], ...] = ((9.0, 15.0),)
    uc_peak_intervals_hours: tuple[tuple[float, float], ...] = ((20.0, 22.0),)
    der_reference_rate_per_kwh: float = 0.30
    dr_incentive_per_kwh: float = 0.50

    def validate(self, system):
        if self.mechanism_version != VERSION:
            raise ValueError(f"Only {VERSION} is implemented; migrate legacy configurations explicitly")
        for name in ("der_bid_cap", "shift_backlog_limit_kwh"):
            personal(getattr(self, name), name, system.num_consumers)
        for name in ("dr_procurement_max_kw", "der_reference_rate_per_kwh", "dr_incentive_per_kwh"):
            nonnegative(getattr(self, name), name)
        if (isinstance(self.shift_deadline_steps, bool) or not isinstance(self.shift_deadline_steps, int)
                or self.shift_deadline_steps < 1):
            raise ValueError("shift_deadline_steps must be a positive integer")
        if list(self.tier_thresholds_kwh) != sorted(set(self.tier_thresholds_kwh)):
            raise ValueError("Tier thresholds must be strictly increasing")
        for x in self.tier_thresholds_kwh:
            nonnegative(x, "tier threshold")
        for name in ("uc_flat_rates", "uc_high_rates", "uc_peak_rates"):
            rates = getattr(self, name)
            if len(rates) != len(self.tier_thresholds_kwh) + 1:
                raise ValueError("Tariff rates and thresholds differ in length")
            for x in rates:
                nonnegative(x, name)
        if any(individual(self.der_bid_cap, i) < self.der_reference_rate_per_kwh
               for i in range(system.num_consumers)):
            raise ValueError("DER bid cap is below DER tariff floor")
        if self.der_reference_rate_per_kwh > min(self.uc_flat_rates + self.uc_high_rates + self.uc_peak_rates):
            raise ValueError("DER tariff floor exceeds an applicable UC tariff")
        tariff_intervals = []
        for name in ("uc_high_intervals_hours", "uc_peak_intervals_hours"):
            for start, end in getattr(self, name):
                nonnegative(start, name)
                nonnegative(end, name)
                if not 0 <= start < end <= 24:
                    raise ValueError("UC tariff intervals must be half-open hour ranges within [0,24]")
                tariff_intervals.append((start, end))
        tariff_intervals.sort()
        if any(a[1] > b[0] for a, b in zip(tariff_intervals, tariff_intervals[1:])):
            raise ValueError("UC high and peak tariff intervals must not overlap")
        for start, end in self.dr_event_intervals:
            if (any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end))
                    or not 0 <= start < end <= system.steps_per_day):
                raise ValueError("DR event intervals must be daily half-open step ranges")
        intervals = sorted(self.dr_event_intervals)
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("DR event intervals overlap")

@dataclass(frozen=True)
class RewardConfig:
    consumer_native_loss_rate_per_kwh: float | tuple[float, ...] = 2.0
    consumer_wait_rate_per_kwh_hour: float | tuple[float, ...] = 0.10
    consumer_overdue_rate_per_kwh_hour: float | tuple[float, ...] = 0.50
    consumer_terminal_pending_rate_per_kwh: float | tuple[float, ...] = 3.0
    actual_compensation_per_kwh: float | tuple[float, ...] = 1.0
    adjustment_rate_per_kwh: float | tuple[float, ...] = 0.50
    uc_supply_shortfall_rate_per_kwh: float = 2.0
    uc_due_rate_per_kwh: float = 1.0
    uc_terminal_pending_rate_per_kwh: float = 2.0
    adjustment_epsilon_kw: float = 1.0
    ess_throughput_rate_per_kwh: float = 0.02
    curtailment_rate_per_kwh: float = 0.10
    der_operating_rate_per_kwh: float = 0.0
    dr_value_rate_per_kwh: float = 0.0
    terminal_energy_value_per_kwh: float = 0.0
    external_cost_per_period: float = 0.0

    def validate(self, count):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.startswith("consumer_") or f.name in {"actual_compensation_per_kwh", "adjustment_rate_per_kwh"}:
                personal(value, f.name, count)
            else:
                nonnegative(value, f.name)
        if self.adjustment_epsilon_kw <= 0:
            raise ValueError("adjustment_epsilon_kw must be positive")

@dataclass(frozen=True)
class EnvironmentConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def validate(self):
        self.system.validate()
        self.market.validate(self.system)
        self.rewards.validate(self.system.num_consumers)
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if set(values) - {"system", "market", "rewards"}:
            raise ValueError("Unknown configuration group")
        result = {}
        for name, kind in (("system", SystemConfig), ("market", MarketConfig), ("rewards", RewardConfig)):
            data = dict(values.get(name, {}))
            unknown = set(data) - {f.name for f in fields(kind)}
            if unknown:
                raise ValueError(f"Unknown or legacy {name} fields: {sorted(unknown)}")
            data = {key: immutable_value(val) for key, val in data.items()}
            result[name] = kind(**data)
        return cls(**result).validate()

    @classmethod
    def from_directory(cls, directory):
        import yaml
        values = {}
        for name in ("system", "market", "rewards"):
            with (Path(directory) / f"{name}.yaml").open(encoding="utf-8") as stream:
                value = yaml.safe_load(stream)
            if not isinstance(value, dict):
                raise ValueError(f"{name}.yaml must contain a mapping")
            values[name] = value
        return cls.from_dict(values)

