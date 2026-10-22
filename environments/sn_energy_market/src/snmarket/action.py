"""Pure policy-output maps and strict physical-action validation (v9)."""

import math
from dataclasses import asdict
from datetime import datetime, timedelta

import numpy as np

from .config import individual
from .contracts import ConsumerAction, LocalBounds, MarketOrder, UCAction
from .tariff import event_context, uc_price

UC_PHYSICAL_FIELDS = tuple(UCAction.__dataclass_fields__)
CONSUMER_PHYSICAL_FIELDS = tuple(ConsumerAction.__dataclass_fields__)


def finite_vector(values, size):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite physical vector of shape ({size},)")
    return array


def action_tuple(action):
    if isinstance(action, (UCAction, ConsumerAction)):
        return tuple(float(v) for v in asdict(action).values())
    array = np.asarray(action, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Actions must be finite vectors")
    return tuple(float(v) for v in array)


def coerce_uc_action(value):
    return UCAction(**dict(zip(UC_PHYSICAL_FIELDS, finite_vector(action_tuple(value), 6), strict=True)))


def coerce_consumer_action(value):
    return ConsumerAction(**dict(zip(CONSUMER_PHYSICAL_FIELDS, finite_vector(action_tuple(value), 4), strict=True)))


def uc_control_bounds(state, generation_kw, config):
    """Decision-time upper bounds in the six physical-coordinate order."""
    sys = config.system
    dt = sys.dt_hours
    charge_cap = max(0.0, min(sys.ess_charge_max_kw,
                     (sys.ess_energy_capacity_kwh - state.ess_energy_kwh)
                     / (sys.ess_charge_efficiency * dt)))
    dis_cap = max(0.0, min(sys.ess_discharge_max_kw,
                  sys.ess_discharge_efficiency * (state.ess_energy_kwh - sys.ess_energy_min_kwh) / dt))
    active, _ = event_context(state.timestamp, config)
    return np.array([sys.grid_import_max_kw, generation_kw, charge_cap, charge_cap,
                     dis_cap, float(active) * config.market.dr_procurement_max_kw], dtype=np.float64)


def map_uc_controls(controls, state, generation_kw, config):
    """§13.2: [mode, six raw controls] -> the original six physical decisions.

    Modes: 0 idle, 1 charge, 2 discharge. Source coordinates directly set
    powers in conditional feasible intervals; there is no signed common gate.
    """
    controls = finite_vector(controls, 7)
    mode = controls[0]
    if mode not in (0, 1, 2):
        raise ValueError("Storage mode must be 0 (idle), 1 (charge), or 2 (discharge)")
    u = np.clip(controls[1:], 0.0, 1.0)
    caps = uc_control_bounds(state, generation_kw, config)
    pg, curtail = u[0] * caps[0], u[1] * caps[1]
    available = (True, caps[2] > 0 and generation_kw - curtail + pg > 0, caps[4] > 0)
    if not available[int(mode)]:
        raise ValueError("Storage mode is unavailable at the decision-time resource state")
    der_charge = u[2] * min(generation_kw - curtail, caps[2]) if mode == 1 else 0.0
    grid_charge = u[3] * min(pg, max(0.0, caps[2] - der_charge)) if mode == 1 else 0.0
    return UCAction(grid_import_offer_kw=float(pg), der_planned_curtailment_kw=float(curtail),
                    der_to_ess_kw=float(der_charge), grid_to_ess_kw=float(grid_charge),
                    ess_discharge_offer_kw=float(u[4] * caps[4]) if mode == 2 else 0.0,
                    dr_procurement_cap_kw=float(u[5] * caps[5]))


def build_local_bounds(consumer, base, rigid, flex, period, config, index=0, timestamp=None):
    values = (base, rigid, flex, consumer.backlog_kwh)
    if not all(math.isfinite(v) and v >= 0.0 for v in values) or rigid > base:
        raise ValueError("Invalid baseline, rigid load, flexibility or backlog")
    limit = individual(config.market.shift_backlog_limit_kwh, index)
    backlog = consumer.backlog_kwh
    if backlog > limit + 1e-9:
        raise ValueError("Queue exceeds its capacity")
    due_energy = math.fsum(t.remaining_kwh for t in consumer.tasks if t.deadline_period <= period)
    dt = config.system.dt_hours
    due = min(due_energy / dt, flex)
    timestamp = timestamp or (datetime.fromisoformat(config.system.start_datetime)
                              + timedelta(hours=period * dt))
    tariff = uc_price(timestamp, consumer.monthly_uc_energy_kwh, config)
    upper_bid = min(individual(config.market.der_bid_cap, index), tariff)
    floor = config.market.der_reference_rate_per_kwh
    if floor > upper_bid:
        raise ValueError("DER reference price exceeds the current legal bid ceiling")
    finite = config.system.episode_mode == "finite"
    allow = not finite or period + config.market.shift_deadline_steps <= config.system.horizon_steps - 1
    return LocalBounds(
        min(base - rigid, flex - due), due, due_energy, backlog, limit,
        float(base), float(rigid), float(flex), dt, allow, floor, upper_bid, tariff,
    )


def validate_consumer_action(value, bounds):
    action = coerce_consumer_action(value)
    c, shift, rho, bid = action_tuple(action)
    tol = 1e-9
    if c < 0.0 or c > bounds.cut_max_kw + tol:
        raise ValueError("Consumer cut violates the due-reserved flexible domain")
    low, high = bounds.shift_interval(c)
    if shift < low - tol or shift > high + tol:
        raise ValueError("Consumer shift violates the conditional local domain")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("DER preference must be in [0,1]")
    if not bounds.der_bid_min <= bid <= bounds.der_bid_max:
        raise ValueError("DER bid is outside its frozen price interval")
    return action


def map_consumer_controls(u, bounds):
    """Four [0,1] policy controls mapped to the document's conditional local domain."""
    values = finite_vector(u, 4)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Consumer policy controls must be in [0,1]")
    cut = float(values[0]) * bounds.cut_max_kw
    low, high = bounds.shift_interval(cut)
    shift = min(high, max(low, low + float(values[1]) * (high - low)))
    bid = min(bounds.der_bid_max, max(bounds.der_bid_min,
              bounds.der_bid_min + float(values[3]) * (bounds.der_bid_max - bounds.der_bid_min)))
    return ConsumerAction(curtail_kw=cut, signed_shift_kw=shift, der_preference_fraction=float(values[2]), der_bid_per_kwh=bid)


def prepare_order(value, consumer, base, rigid, flex, period, config, index=0, timestamp=None):
    bounds = build_local_bounds(consumer, base, rigid, flex, period, config, index, timestamp)
    action = validate_consumer_action(value, bounds)
    native = base - action.curtail_kw - max(action.signed_shift_kw, 0.0)
    request = base - action.curtail_kw - action.signed_shift_kw
    return MarketOrder(consumer.consumer_id, index, action, base, rigid, flex,
                       native, request, native + bounds.due_kw, bounds.due_kw,
                       bounds.due_kwh, bounds.uc_tariff)

