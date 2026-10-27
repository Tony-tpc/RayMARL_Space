"""Two-stage v9 engine: validate complete candidates, then commit once."""
import copy
import hashlib
import json
import math
import threading
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from uuid import uuid4

import numpy as np

from .action import action_tuple, build_local_bounds, prepare_order
from .config import EnvironmentConfig, individual, nonnegative
from .consumer import next_consumer_state
from .contracts import (
    ConsumerState,
    EngineSnapshot,
    ExecutionRecord,
    MarketState,
    Phase,
    ShiftTask,
    StateDelta,
)
from .dr import realize_service
from .incentive import procure
from .market import clear
from .observation import build_consumer_observation, build_uc_observation
from .physics import execute
from .profiles import Profiles, synthetic_profiles
from .reward import calculate_rewards
from .settlement import settle
from .uc import prepare_offer


class StageError(RuntimeError):
    pass

class ProfileExhaustedError(RuntimeError):
    pass

class EnergyMarketEngine:
    def __init__(self, config=None, profiles=None):
        self.config = EnvironmentConfig.from_dict((config or EnvironmentConfig()).to_dict())
        self.profiles = profiles if profiles is not None else synthetic_profiles(self.config)
        s = self.config.system
        if not isinstance(self.profiles, Profiles) or self.profiles.loads_kw.shape[1] != s.num_consumers:
            raise ValueError("Profiles must match the configured consumers")
        if s.episode_mode == "finite" and len(self.profiles) != s.horizon_steps:
            raise ValueError("Finite profiles must match horizon_steps")
        configured_start = datetime.fromisoformat(s.start_datetime)
        if (self.profiles.resolution_minutes != s.resolution_minutes
                or self.profiles.start_datetime != configured_start
                or self.profiles.start_datetime.utcoffset() != configured_start.utcoffset()):
            raise ValueError("Profile and configured calendar differ")
        self.consumer_ids = tuple(f"consumer_{i}" for i in range(s.num_consumers))
        self.compatibility_signature = self._signature(self.profiles)
        self._lock = threading.RLock()
        self._state = None
        self._phase = Phase.UC_TURN
        self._pending_plan = self._pending_offer = self._pending_raw_uc = None
        self._candidate_signature = None
        self._temporary_announcement_active = False
        self._records = {}
        self._payment_keys = set()
        self._rng = np.random.default_rng()

    def _signature(self, profiles):
        return hashlib.sha256((json.dumps(self.config.to_dict(), sort_keys=True)
                               + profiles.fingerprint()).encode()).hexdigest()

    @property
    def phase(self):
        with self._lock:
            return self._phase

    @property
    def state(self):
        with self._lock:
            self._require_reset()
            # All state leaves are immutable scalars / tuples / frozen dataclasses.
            return self._state

    @property
    def records(self):
        with self._lock:
            return copy.deepcopy(tuple(self._records.values()))

    @property
    def latest_record(self):
        with self._lock:
            return copy.deepcopy(next(reversed(self._records.values()))) if self._records else None

    @property
    def data_exhausted(self):
        return self._state is not None and self._state.period_id >= len(self.profiles)

    def _require_reset(self):
        if self._state is None:
            raise StageError("Call reset before interacting with the environment")

    def _require_phase(self, phase):
        self._require_reset()
        if self._phase != phase:
            raise StageError(f"Expected {phase}, current phase is {self._phase}")

    def _require_data(self):
        if self.data_exhausted:
            raise ProfileExhaustedError("Exogenous data exhausted; append a contiguous profile block")

    def _validate_state(self, state):
        s, m = self.config.system, self.config.market
        if (not isinstance(state, MarketState) or not isinstance(state.period_id, int)
                or isinstance(state.period_id, bool) or not 0 <= state.period_id <= len(self.profiles)
                or state.timestamp != self.profiles.timestamp(state.period_id)
                or state.timestamp.utcoffset() != self.profiles.start_datetime.utcoffset()):
            raise ValueError("Invalid state period/calendar")
        if not isinstance(state.episode_id, str) or not state.episode_id:
            raise ValueError("episode_id must be a nonempty string")
        if not isinstance(state.consumers, tuple) or not isinstance(state.last_totals, tuple):
            raise ValueError("State consumer and historical feedback collections must be immutable tuples")
        for energy in (state.ess_energy_kwh, state.initial_ess_energy_kwh):
            nonnegative(energy, "ESS energy")
            if not s.ess_energy_min_kwh - 1e-8 <= energy <= s.ess_energy_capacity_kwh + 1e-8:
                raise ValueError("ESS energy violates bounds")
        if tuple(c.consumer_id for c in state.consumers) != self.consumer_ids:
            raise ValueError("Consumer IDs/order do not match configuration")
        if len(state.last_totals) != 6 or not all(math.isfinite(v) for v in state.last_totals):
            raise ValueError("Invalid historical feedback")
        if any(v < -1e-8 for v in state.last_totals[:5]):
            raise ValueError("Historical physical totals cannot be negative")
        for i, c in enumerate(state.consumers):
            if not isinstance(c, ConsumerState) or not isinstance(c.tasks, tuple):
                raise ValueError("State requires immutable consumer/task tuples")
            nonnegative(c.monthly_uc_energy_kwh, "monthly energy")
            if c.backlog_kwh > individual(m.shift_backlog_limit_kwh, i) + 1e-8:
                raise ValueError("Queue exceeds its capacity")
            task_ids = set()
            for task in c.tasks:
                if (not isinstance(task, ShiftTask)
                        or any(isinstance(x, bool) or not isinstance(x, int)
                               for x in (task.source_period, task.deadline_period))
                        or task.source_period >= state.period_id
                        or task.deadline_period <= task.source_period
                        or task.deadline_period > state.period_id + m.shift_deadline_steps):
                    raise ValueError("Invalid task source/deadline or deadline bucket window")
                nonnegative(task.remaining_kwh, "task energy")
                if not isinstance(task.task_id, str) or not task.task_id.strip() or task.task_id in task_ids:
                    raise ValueError("Task IDs must be nonempty strings and unique per consumer")
                task_ids.add(task.task_id)

    @staticmethod
    def _initial_tasks(consumer):
        """Assign deterministic provenance IDs without changing task order or energy."""
        tasks = tuple(consumer.tasks)
        if not all(isinstance(task, ShiftTask) for task in tasks):
            raise ValueError("Initial queues must contain ShiftTask objects")
        used = {task.task_id for task in tasks if isinstance(task.task_id, str)}
        normalized = []
        for ordinal, task in enumerate(tasks):
            if task.task_id is None:
                base = f"{consumer.consumer_id}:initial:{task.source_period}:{ordinal}"
                task_id, suffix = base, 0
                while task_id in used:
                    suffix += 1
                    task_id = f"{base}:{suffix}"
                used.add(task_id)
                task = replace(task, task_id=task_id)
            normalized.append(task)
        return tuple(normalized)

    def reset(self, *, seed=None, options=None):
        with self._lock:
            options = options or {}
            if set(options) - {"episode_id", "consumer_states", "ess_energy_kwh"}:
                raise ValueError("Unknown reset options")
            initial = options.get("consumer_states", tuple(ConsumerState(i) for i in self.consumer_ids))
            consumers = tuple(ConsumerState(c.consumer_id, float(c.monthly_uc_energy_kwh),
                                            self._initial_tasks(c)) for c in initial)
            energy = float(options.get("ess_energy_kwh", self.config.system.ess_initial_energy_kwh))
            state = MarketState(options.get("episode_id", uuid4().hex), 0, self.profiles.timestamp(0),
                                energy, consumers, initial_ess_energy_kwh=energy)
            self._validate_state(state)
            observation = build_uc_observation(state, self.profiles, self.config)
            rng = np.random.default_rng(seed)
            self._state, self._phase, self._rng = state, Phase.UC_TURN, rng
            self._pending_plan = self._pending_offer = self._pending_raw_uc = self._candidate_signature = None
            self._records, self._payment_keys = {}, set()
            return observation

    def uc_observation(self):
        with self._lock:
            self._require_reset()
            return build_uc_observation(self._state, self.profiles, self.config)

    @contextmanager
    def temporary_announcement(self):
        """Permit a read-only UC announcement without copying committed history."""
        with self._lock:
            self._require_phase(Phase.UC_TURN)
            state, records, payments = self._state, len(self._records), len(self._payment_keys)
            pending = (self._phase, self._pending_plan, self._pending_offer,
                       self._pending_raw_uc, self._candidate_signature)
            rng = copy.deepcopy(self._rng.bit_generator.state)
            self._temporary_announcement_active = True
            try:
                yield
            finally:
                self._temporary_announcement_active = False
                # Publishing may change only pending decision fields, never commit a period.
                if self._state is not state or len(self._records) != records or len(self._payment_keys) != payments:
                    raise StageError("A temporary announcement must not execute a physical period")
                (self._phase, self._pending_plan, self._pending_offer,
                 self._pending_raw_uc, self._candidate_signature) = pending
                self._rng.bit_generator.state = rng

    def _consumer_observations(self, offer, state):
        index = min(state.period_id, len(self.profiles) - 1)
        p = self.profiles
        return {c.consumer_id: build_consumer_observation(
            state, c, p.loads_kw[index, i], p.rigid_kw[index, i], p.flex_kw[index, i],
            p.der_kw[index], offer, self.config, i) for i, c in enumerate(state.consumers)}

    def consumer_observations(self, *, terminal=False):
        with self._lock:
            if terminal:
                self._require_phase(Phase.TERMINATED)
                # Final carrier: last real announcement + final state, no fictitious new action.
                offer = next(reversed(self._records.values())).offer
            else:
                self._require_phase(Phase.CONSUMER_TURN)
                offer = self._pending_offer
            return self._consumer_observations(offer, self._state)

    def local_bounds(self):
        with self._lock:
            self._require_phase(Phase.CONSUMER_TURN)
            p, t, state = self.profiles, self._state.period_id, self._state
            return {c.consumer_id: build_local_bounds(c, p.loads_kw[t, i], p.rigid_kw[t, i],
                    p.flex_kw[t, i], t, self.config, i, state.timestamp)
                    for i, c in enumerate(state.consumers)}

    def apply_uc_action(self, action):
        with self._lock:
            self._require_phase(Phase.UC_TURN)
            self._require_data()
            der = self.profiles.der_kw[self._state.period_id]
            offer = prepare_offer(action, self._state, der, self.config)
            plan = offer.action
            observations = self._consumer_observations(offer, self._state)
            self._pending_plan, self._pending_offer = plan, offer
            self._pending_raw_uc = action_tuple(plan)
            self._candidate_signature = None
            self._phase = Phase.CONSUMER_TURN
            return observations

    @staticmethod
    def _input_signature(uc_action, actions):
        payload = [action_tuple(uc_action), [(key, action_tuple(actions[key])) for key in sorted(actions)]]
        return hashlib.sha256(repr(payload).encode()).hexdigest()

    @staticmethod
    def _delta_signature(delta):
        return hashlib.sha256(repr(delta).encode()).hexdigest()

    def evaluate_consumer_actions(self, actions, *, transition_id=None):
        with self._lock:
            self._require_phase(Phase.CONSUMER_TURN)
            if set(actions) != set(self.consumer_ids):
                raise ValueError("Provide exactly one action per consumer")
            if transition_id is None:
                transition_id = f"{self._state.episode_id}:{self._state.period_id}"
            if not isinstance(transition_id, str) or not transition_id or transition_id in self._records:
                raise ValueError("Transition ID must be a new nonempty string")
            state, config, p = self._state, self.config, self.profiles
            t, dt = state.period_id, config.system.dt_hours
            orders = tuple(prepare_order(actions[c.consumer_id], c, p.loads_kw[t, i], p.rigid_kw[t, i],
                         p.flex_kw[t, i], t, config, i, state.timestamp) for i, c in enumerate(state.consumers))
            market = clear(orders, self._pending_offer)
            physical = execute(self._pending_offer, market, p.der_kw[t], state, config)
            services = {o.consumer_id: realize_service(o, market.allocations[o.consumer_id], c, t, config, i)
                        for i, (o, c) in enumerate(zip(orders, state.consumers, strict=True))}
            dr = procure(orders, self._pending_offer, dt)
            terminated = config.system.episode_mode == "finite" and t + 1 == config.system.horizon_steps
            ledger = settle(orders, services, market, physical, dr, self._pending_offer, state,
                            p.grid_cost_per_kwh[t], terminated, config)
            rewards = calculate_rewards(ledger)
            if not all(math.isfinite(v) for v in rewards.values()):
                raise ValueError("Nonfinite rewards")
            timestamp = p.timestamp(t + 1)
            rolled = (timestamp.year, timestamp.month) != (state.timestamp.year, state.timestamp.month)
            consumers, closing = [], {}
            for c in state.consumers:
                after_c, total = next_consumer_state(c, market.allocations[c.consumer_id],
                                                     services[c.consumer_id], rolled, dt)
                consumers.append(after_c)
                closing[c.consumer_id] = total
            totals = (market.der_total_kw, market.uc_total_kw, dr.net_dr_kw, dr.total_payment,
                      math.fsum(s.native_lost_kw for s in services.values()), ledger.cash_profit)
            after = MarketState(state.episode_id, t + 1, timestamp, physical.ess_energy_after_kwh,
                                tuple(consumers), totals, state.initial_ess_energy_kwh)
            self._validate_state(after)
            # A failed next-observation build must not leave a half committed physical step.
            build_uc_observation(after, p, config)
            if terminated:
                self._consumer_observations(self._pending_offer, after)
            self._validate_results(orders, market, physical, services, ledger)
            record = ExecutionRecord(transition_id, state.episode_id, t, state.timestamp,
                config.market.mechanism_version, self._input_signature(self._pending_plan, actions),
                self._pending_plan, {o.consumer_id: o.action for o in orders}, orders, self._pending_offer,
                market, physical, services, dr, ledger, rewards, closing, rolled, terminated,
                decision_state=state,
                uc_observation=tuple(float(x) for x in build_uc_observation(state, p, config)),
                consumer_observations={key: tuple(float(x) for x in value)
                                       for key, value in self._consumer_observations(self._pending_offer, state).items()},
                local_bounds=self.local_bounds(),
                supply_diagnostics={
                    "physical_max_supply_kw": float(p.der_kw[t] + config.system.grid_import_max_kw
                                                    + self._pending_offer.discharge_available_kw),
                    "grid_capacity_withheld_kw": config.system.grid_import_max_kw - self._pending_plan.grid_import_offer_kw,
                    "discharge_capacity_withheld_kw": self._pending_offer.discharge_available_kw - self._pending_plan.ess_discharge_offer_kw,
                    "charging_diversion_kw": self._pending_plan.der_to_ess_kw + self._pending_plan.grid_to_ess_kw,
                    "active_curtailment_kw": self._pending_plan.der_planned_curtailment_kw,
                })
            keys = tuple((state.episode_id, t, i, self._pending_offer.event_id) for i in self.consumer_ids)
            delta = StateDelta(state, after, record, keys)
            self._candidate_signature = self._delta_signature(delta)
            return delta

    def _validate_results(self, orders, market, physical, services, ledger):
        scale = max(1.0, sum(o.request_kw for o in orders), physical.der_generation_kw, physical.grid_buy_kw)
        tol = 1e-8 + 1e-10 * scale
        residuals = [physical.power_residual_kw, ledger.cash_residual]
        residuals += [v for s in services.values() for v in
                      (s.request_residual_kw, s.original_load_residual_kw, s.queue_residual_kwh)]
        expected = min(sum(o.request_kw for o in orders),
                       self._pending_offer.der_capacity_kw + self._pending_offer.uc_capacity_kw)
        residuals.append(market.der_total_kw + market.uc_total_kw - expected)
        if not all(math.isfinite(v) and abs(v) <= tol for v in residuals):
            raise ValueError(f"Energy, cash, queue or market conservation failed: {residuals}, tolerance={tol}")

    def commit(self, delta):
        with self._lock:
            if self._temporary_announcement_active:
                raise StageError("A temporary announcement cannot commit a physical period")
            if not isinstance(delta, StateDelta):
                raise ValueError("Expected a StateDelta")
            identity = delta.record.transition_id
            if identity in self._records:
                if self._records[identity] != delta.record:
                    raise ValueError("Transition ID already belongs to different results")
                return copy.deepcopy(self._records[identity])
            self._require_phase(Phase.CONSUMER_TURN)
            if (delta.before != self._state or delta.after.period_id != self._state.period_id + 1
                    or self._delta_signature(delta) != self._candidate_signature):
                raise ValueError("Stale or modified StateDelta")
            if self._payment_keys.intersection(delta.payment_keys):
                raise ValueError("Duplicate payment key")
            stored = copy.deepcopy(delta.record)
            # Prepare copies before changing committed state; append histories in O(1) per step.
            self._records[identity] = stored
            self._payment_keys.update(delta.payment_keys)
            self._state = delta.after
            self._phase = Phase.TERMINATED if stored.terminated else Phase.UC_TURN
            self._pending_plan = self._pending_offer = self._pending_raw_uc = self._candidate_signature = None
            return delta.record

    def apply_consumer_actions(self, actions, *, transition_id=None):
        with self._lock:
            if transition_id is not None and transition_id in self._records:
                old = self._records[transition_id]
                if self._input_signature(old.uc_action, actions) != old.input_signature:
                    raise ValueError("Transition replay has different consumer actions")
                return copy.deepcopy(old)
            return self.commit(self.evaluate_consumer_actions(actions, transition_id=transition_id))

    def run_period(self, uc_action, consumer_actions, *, transition_id=None):
        with self._lock:
            if transition_id is not None and transition_id in self._records:
                old = self._records[transition_id]
                if self._input_signature(uc_action, consumer_actions) != old.input_signature:
                    raise ValueError("Transition replay has different inputs")
                return copy.deepcopy(old)
            self.apply_uc_action(uc_action)
            return self.apply_consumer_actions(consumer_actions, transition_id=transition_id)

    def get_state(self):
        with self._lock:
            self._require_reset()
            return copy.deepcopy(EngineSnapshot(self.compatibility_signature, self._state, self._phase,
                self._pending_plan, self._pending_offer, self._pending_raw_uc, self._records,
                frozenset(self._payment_keys), self._rng.bit_generator.state))

    def _branch_state(self):
        """Trusted in-process branch: committed records are never modified internally.

        The public get_state()/set_state() continue to deep-copy and validate all
        history. This private path copies the index, sharing only stored records;
        commit() inserts new records and every public record accessor returns a copy.
        """
        with self._lock:
            self._require_reset()
            if self._phase == Phase.CONSUMER_TURN:
                raise StageError("Branch snapshots require a completed physical period")
            return EngineSnapshot(self.compatibility_signature, self._state, self._phase,
                None, None, None, self._records.copy(), frozenset(self._payment_keys),
                copy.deepcopy(self._rng.bit_generator.state))

    def _restore_branch(self, snapshot):
        with self._lock:
            if snapshot.compatibility_signature != self.compatibility_signature:
                raise ValueError("Branch snapshot belongs to a different scenario")
            self._state, self._phase = snapshot.state, snapshot.phase
            self._pending_plan = self._pending_offer = self._pending_raw_uc = self._candidate_signature = None
            self._records = snapshot.records.copy()
            self._payment_keys = set(snapshot.payment_keys)
            self._rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)

    def set_state(self, snapshot):
        with self._lock:
            if not isinstance(snapshot, EngineSnapshot) or snapshot.compatibility_signature != self.compatibility_signature:
                raise ValueError("Snapshot has a different rule/configuration/profile version")
            snap = copy.deepcopy(snapshot)
            self._validate_state(snap.state)
            if not isinstance(snap.phase, Phase):
                raise ValueError("Invalid snapshot phase")
            terminal = self.config.system.episode_mode == "finite" and snap.state.period_id == self.config.system.horizon_steps
            if (snap.phase == Phase.TERMINATED) != terminal:
                raise ValueError("Snapshot terminal phase/time mismatch")
            if (len(snap.records) != snap.state.period_id
                    or {r.period_id for r in snap.records.values()} != set(range(snap.state.period_id))
                    or any(k != r.transition_id or r.episode_id != snap.state.episode_id
                           or r.mechanism_version != self.config.market.mechanism_version
                           for k, r in snap.records.items())):
                raise ValueError("Snapshot execution history is incomplete/inconsistent")
            keys = frozenset((snap.state.episode_id, r.period_id, i, r.offer.event_id)
                             for r in snap.records.values() for i in self.consumer_ids)
            if keys != snap.payment_keys:
                raise ValueError("Snapshot payment keys do not match history")
            if snap.records:
                first = min(snap.records.values(), key=lambda r: r.period_id).physical
                s = self.config.system
                initial = (first.ess_energy_after_kwh - s.ess_charge_efficiency * first.ess_charge_kw * s.dt_hours
                           + first.ess_discharge_kw * s.dt_hours / s.ess_discharge_efficiency)
                if not math.isclose(initial, snap.state.initial_ess_energy_kwh, abs_tol=1e-8, rel_tol=1e-10):
                    raise ValueError("Snapshot initial energy differs from first execution")
                last = max(snap.records.values(), key=lambda r: r.period_id)
                feedback = (last.market.der_total_kw, last.market.uc_total_kw, last.dr.net_dr_kw,
                            last.dr.total_payment, math.fsum(x.native_lost_kw for x in last.services.values()),
                            last.ledger.cash_profit)
                if snap.state.last_totals != feedback:
                    raise ValueError("Snapshot historical feedback differs from last execution")
                if last.physical.ess_energy_after_kwh != snap.state.ess_energy_kwh:
                    raise ValueError("Snapshot SOC does not match its last execution")
                for c in snap.state.consumers:
                    if (c.tasks != last.services[c.consumer_id].tasks_after
                            or c.monthly_uc_energy_kwh != (0 if last.month_rolled
                                else last.monthly_closing_energy_kwh[c.consumer_id])):
                        raise ValueError("Snapshot consumer state does not match its last execution")
            elif (snap.state.initial_ess_energy_kwh != snap.state.ess_energy_kwh
                  or snap.state.last_totals != (0.0,) * 6):
                raise ValueError("Initial snapshot has inconsistent initial energy or feedback")
            if snap.phase == Phase.CONSUMER_TURN:
                if snap.pending_raw_uc is None or snap.state.period_id >= len(self.profiles):
                    raise ValueError("Consumer snapshot missing its valid announcement/data")
                offer = prepare_offer(snap.pending_raw_uc, snap.state,
                                      self.profiles.der_kw[snap.state.period_id], self.config)
                if offer != snap.pending_offer or offer.action != snap.pending_plan:
                    raise ValueError("Snapshot announcement inconsistent with its action")
            elif any(v is not None for v in (snap.pending_plan, snap.pending_offer, snap.pending_raw_uc)):
                raise ValueError("Non-consumer snapshot contains pending action")
            build_uc_observation(snap.state, self.profiles, self.config)
            rng = np.random.default_rng()
            rng.bit_generator.state = snap.rng_state
            self._state, self._phase = snap.state, snap.phase
            self._pending_plan, self._pending_offer, self._pending_raw_uc = (
                snap.pending_plan, snap.pending_offer, snap.pending_raw_uc)
            # Snapshot mappings may be deserialized in a different insertion order.
            # Public history and the terminal announcement follow physical time.
            records = dict(sorted(snap.records.items(), key=lambda item: item[1].period_id))
            self._records, self._payment_keys, self._rng = records, set(snap.payment_keys), rng
            self._candidate_signature = None

    def append_profiles(self, profiles):
        with self._lock:
            self._require_phase(Phase.UC_TURN)
            if self.config.system.episode_mode != "continuous":
                raise ValueError("Only continuous environments accept appended data")
            updated = self.profiles.append(profiles)
            signature = self._signature(updated)
            build_uc_observation(self._state, updated, self.config)
            self.profiles, self.compatibility_signature = updated, signature

    def central_state(self):
        """Trusted diagnostics/consumer training state; never a UC observation payload."""
        with self._lock:
            self._require_reset()
            t = min(self._state.period_id, len(self.profiles) - 1)
            return {"phase": self._phase.value, "state": asdict(self._state),
                    "baseline_kw": self.profiles.loads_kw[t].tolist(),
                    "rigid_kw": self.profiles.rigid_kw[t].tolist(),
                    "flex_kw": self.profiles.flex_kw[t].tolist(),
                    "der_generation_kw": float(self.profiles.der_kw[t]),
                    "grid_cost_per_kwh": float(self.profiles.grid_cost_per_kwh[t]),
                    "public_offer": asdict(self._pending_offer) if self._pending_offer else None}

