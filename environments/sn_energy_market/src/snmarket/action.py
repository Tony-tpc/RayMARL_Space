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


def map_uc_controls(z, state, generation_kw, config):
    """Document §13.2 output layer. The engine never calls this to repair an action."""
    z = finite_vector(z, 6)
    u = np.clip(z, 0.0, 1.0)
    signed = float(np.clip(z[4], -1.0, 1.0))
    sys = config.system
    dt = sys.dt_hours
    pg = float(u[0]) * sys.grid_import_max_kw
    curtail = float(u[1]) * generation_kw
    der_int = float(u[2]) * max(signed, 0.0) * (generation_kw - curtail)
    pg_int = float(u[3]) * max(signed, 0.0) * pg
    charge_cap = min(sys.ess_charge_max_kw,
                     (sys.ess_energy_capacity_kwh - state.ess_energy_kwh)
                     / (sys.ess_charge_efficiency * dt))
    total = der_int + pg_int
    beta = min(1.0, max(0.0, charge_cap) / total) if total > 0.0 else 0.0
    dis_cap = min(sys.ess_discharge_max_kw,
                  sys.ess_discharge_efficiency * (state.ess_energy_kwh - sys.ess_energy_min_kwh) / dt)
    active, _ = event_context(state.timestamp, config)
    return UCAction(grid_import_offer_kw=pg, der_planned_curtailment_kw=curtail, der_to_ess_kw=beta * der_int, grid_to_ess_kw=beta * pg_int,
                      ess_discharge_offer_kw=max(-signed, 0.0) * max(0.0, dis_cap),
                      dr_procurement_cap_kw=float(u[5]) * float(active) * config.market.dr_procurement_max_kw)


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

