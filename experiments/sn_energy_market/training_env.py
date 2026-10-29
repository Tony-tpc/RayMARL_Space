"""Two decision stages exposed to learning without changing market execution."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, fields

import numpy as np
from snmarket import (
    EnergyMarketEngine,
    EnvironmentConfig,
    Phase,
    ProfileExhaustedError,
    Profiles,
)
from snmarket.action import action_tuple, map_uc_controls, uc_control_bounds
from snmarket.contracts import EngineSnapshot, LocalBounds
from snmarket.observation import consumer_fields, uc_fields

BOUND_FIELDS = tuple(field.name for field in fields(LocalBounds))


@dataclass(frozen=True)
class Decision:
    uc_observation: np.ndarray
    uc_value_state: np.ndarray
    raw_controls: np.ndarray
    uc_action: np.ndarray
    mapping_context: dict[str, float]
    consumer_observations: dict[str, np.ndarray]
    consumer_bounds: dict[str, np.ndarray]
    consumer_value_state: np.ndarray
    period_id: int


@dataclass(frozen=True)
class PeriodResult:
    rewards: dict[str, float]
    terminated: bool
    next_uc_observation: np.ndarray | None
    metrics: dict[str, float]
    data_exhausted: bool


@dataclass(frozen=True)
class MarketSnapshot:
    profiles: Profiles
    engine: EngineSnapshot


class MarketTrainingEnv:
    """The collector owns pending policy samples; this adapter owns physical state.

    ``publish`` creates the Consumer decision state after the UC announcement.
    ``step`` advances one physical period. A subsequent ``publish`` is required
    before evaluating the next Consumer value, including at a sampling cut.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        profiles: Profiles | None = None,
    ):
        self.engine = EnergyMarketEngine(config, profiles)
        self.config = self.engine.config
        self.episode_mode = config.system.episode_mode
        self.consumer_ids = self.engine.consumer_ids
        horizon = config.market.shift_deadline_steps
        self.uc_observation_fields = uc_fields(horizon)
        self.consumer_observation_fields = consumer_fields(horizon)
        self.uc_observation_dim = len(self.uc_observation_fields)
        self.uc_value_dim = self.uc_observation_dim + 12
        self.uc_soc_input = {
            "index": self.uc_observation_fields.index("ess_energy_kwh"),
            "min": config.system.ess_energy_min_kwh,
            "max": config.system.ess_energy_capacity_kwh,
        }
        self.consumer_observation_dim = len(self.consumer_observation_fields)
        self.consumer_value_dim = (
            self.uc_observation_dim
            + len(self.consumer_ids) * self.consumer_observation_dim
        )

    def reset(self, seed: int, *, options: dict | None = None) -> np.ndarray:
        return self.engine.reset(
            seed=seed, options={"episode_id": f"scenario-{seed}", **(options or {})}
        )

    def uc_bounds(self) -> np.ndarray:
        state = self.engine.state
        return uc_control_bounds(state, float(self.engine.profiles.der_kw[state.period_id]), self.config)

    def publish(self, raw_controls) -> Decision:
        if self.engine.data_exhausted:
            raise ProfileExhaustedError(
                "Append contiguous profiles before the next UC decision"
            )
        state, system = self.engine.state, self.config.system
        observation = self.engine.uc_observation()
        # Public decision-time calendar only; no individual Consumer state.
        month = np.zeros(12, dtype=observation.dtype)
        month[state.timestamp.month - 1] = 1.0
        uc_value_state = np.concatenate([observation, month])
        generation = float(self.engine.profiles.der_kw[state.period_id])
        action = map_uc_controls(raw_controls, state, generation, self.config)
        consumer_observations = self.engine.apply_uc_action(action)
        bounds = {
            key: np.asarray(
                [getattr(value, name) for name in BOUND_FIELDS], dtype=np.float64
            )
            for key, value in self.engine.local_bounds().items()
        }
        # Fixed-order, post-announcement state is passed only to Consumer value networks.
        value_state = np.concatenate(
            [observation, *[consumer_observations[key] for key in self.consumer_ids]]
        )
        context = {
            "ess_energy_kwh": state.ess_energy_kwh,
            "der_generation_kw": generation,
            "grid_import_max_kw": system.grid_import_max_kw,
            "ess_energy_min_kwh": system.ess_energy_min_kwh,
            "ess_energy_capacity_kwh": system.ess_energy_capacity_kwh,
            "ess_charge_max_kw": system.ess_charge_max_kw,
            "ess_discharge_max_kw": system.ess_discharge_max_kw,
            "ess_charge_efficiency": system.ess_charge_efficiency,
            "ess_discharge_efficiency": system.ess_discharge_efficiency,
            "dt_hours": system.dt_hours,
            "dr_procurement_max_kw": self.config.market.dr_procurement_max_kw,
            "event_active": float(observation[6]),
        }
        return Decision(
            observation,
            uc_value_state,
            np.asarray(raw_controls, dtype=np.float64).copy(),
            np.asarray(action_tuple(action)),
            context,
            consumer_observations,
            bounds,
            value_state,
            state.period_id,
        )

    def step(self, consumer_actions) -> PeriodResult:
        record = self.engine.apply_consumer_actions(consumer_actions)
        dt = self.config.system.dt_hours
        metrics = {
            "terminal_energy_value": record.ledger.terminal_energy_value,
            "initial_energy_kwh": self.engine.state.initial_ess_energy_kwh if record.terminated else 0.0,
            "final_energy_kwh": record.physical.ess_energy_after_kwh if record.terminated else 0.0,
            "uc_cash": record.ledger.cash_profit,
            "uc_book_profit": record.ledger.book_profit,
            "native_lost_kwh": sum(
                service.native_lost_kw for service in record.services.values()
            )
            * dt,
            "native_requested_kwh": sum(order.native_kw for order in record.orders)
            * dt,
            "served_kwh": record.physical.supply_served_kw * dt,
            "grid_kwh": record.physical.grid_buy_kw * dt,
            "der_charge_kwh": record.uc_action.der_to_ess_kw * dt,
            "grid_charge_kwh": record.uc_action.grid_to_ess_kw * dt,
            "ess_discharge_kwh": record.physical.ess_discharge_kw * dt,
            "der_spill_kwh": record.physical.der_spill_kw * dt,
            "dr_net_kwh": record.dr.net_dr_kw * dt,
            "dr_payment": record.dr.total_payment,
            "supply_shortfall_kwh": record.ledger.supply_shortfall_kw * dt,
            "consumer_cost": sum(
                ledger.total_cost for ledger in record.ledger.consumers.values()
            ),
            "physical_periods": 1.0,
        }
        metrics.update(
            {
                f"consumer_cost/{key}": ledger.total_cost
                for key, ledger in record.ledger.consumers.items()
            }
        )
        next_observation = None
        if not record.terminated and not self.engine.data_exhausted:
            next_observation = self.engine.uc_observation()
        return PeriodResult(
            dict(record.rewards),
            record.terminated,
            next_observation,
            metrics,
            self.engine.data_exhausted and not record.terminated,
        )

    def get_state(self) -> MarketSnapshot:
        return MarketSnapshot(self.engine.profiles, self.engine.get_state())

    def temporary_announcement(self):
        return self.engine.temporary_announcement()

    def branch_state(self):
        return MarketSnapshot(self.engine.profiles, self.engine._branch_state())

    def restore_branch(self, snapshot):
        engine = EnergyMarketEngine(self.config, snapshot.profiles)
        engine._restore_branch(snapshot.engine)
        self.engine = engine

    def set_state(self, snapshot: MarketSnapshot) -> None:
        engine = EnergyMarketEngine(self.config, snapshot.profiles)
        engine.set_state(snapshot.engine)
        self.engine = engine

    def append_profiles(self, profiles: Profiles) -> np.ndarray:
        self.engine.append_profiles(profiles)
        return self.engine.uc_observation()

    @property
    def phase(self) -> Phase:
        return self.engine.phase


def scenario_profiles(
    config: EnvironmentConfig, seed: int, split: str, *, periods: int | None = None
) -> Profiles:
    """Bounded synthetic scenarios; one seed fixes the complete exogenous path."""
    from datetime import datetime

    system = config.system
    rng = np.random.default_rng(seed)
    start = datetime.fromisoformat(system.start_datetime)
    count = periods or system.horizon_steps
    hours = np.arange(count) * system.dt_hours + start.hour + start.minute / 60
    ids = np.arange(system.num_consumers)
    native = (
        5 + ids[None, :] * 0.7 + 2 * (1 + np.sin(2 * np.pi * (hours[:, None] - 8) / 24))
    )
    loads = native * rng.uniform(0.85, 1.15, size=(1, system.num_consumers))
    loads = loads * rng.uniform(0.95, 1.05, size=loads.shape)
    der = (
        system.num_consumers * 5 * np.maximum(0, np.sin(np.pi * (hours % 24 - 6) / 12))
    )
    der = der * rng.uniform(0.75, 1.15, size=count)
    prices = np.where((hours % 24 >= 17) & (hours % 24 < 22), 0.65, 0.35)
    prices = prices * rng.uniform(0.9, 1.1, size=count)
    flex = loads * system.dr_limit_ratio
    return Profiles(
        loads,
        der,
        prices,
        flex,
        rigid_kw=loads - flex,
        start_datetime=system.start_datetime,
        resolution_minutes=system.resolution_minutes,
        metadata={
            "source": "bounded_synthetic_training",
            "value_kind": "realized",
            "seed": seed,
            "split": split,
        },
    )


@dataclass(frozen=True)
class MarketEnvFactory:
    config: EnvironmentConfig
    split: str = "train"
    profile_periods: int | None = None
    seed_splits: dict[str, tuple[int, ...]] | None = None

    def __call__(self, seed: int) -> MarketTrainingEnv:
        split = next(
            (name for name, seeds in (self.seed_splits or {}).items() if seed in seeds),
            self.split,
        )
        return MarketTrainingEnv(
            self.config,
            scenario_profiles(self.config, seed, split, periods=self.profile_periods),
        )

    def get_contract(self) -> dict:
        """Serializable scenario semantics bound to a resumable experiment."""
        contract = {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "config": self.config.to_dict(),
            "split": self.split,
            "profile_periods": self.profile_periods,
            "seed_splits": self.seed_splits,
            "profile_rule_sha256": hashlib.sha256(
                inspect.getsource(scenario_profiles).encode()
            ).hexdigest(),
        }
        return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))
