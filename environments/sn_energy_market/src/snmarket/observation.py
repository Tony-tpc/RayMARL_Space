"""Actor observations in fixed v9 order, physical units, with no private UC fields."""
import numpy as np

from .config import individual
from .tariff import event_context, uc_price

UC_CURRENT_FIELDS = ("time_index", "remaining_steps", "ess_energy_kwh", "der_generation_kw",
    "grid_import_max_kw", "grid_cost_per_kwh", "event_active", "dr_price_per_kwh",
    "aggregate_baseline_kw", "aggregate_rigid_kw", "aggregate_flex_kw", "aggregate_due_kw",
    "weighted_uc_tariff")
UC_HISTORY_FIELDS = ("previous_der_sold_kw", "previous_uc_sold_kw", "previous_net_dr_kw",
    "previous_dr_payment", "previous_native_lost_kw", "previous_cash_profit")
CONSUMER_CURRENT_FIELDS = ("time_index", "remaining_steps", "baseline_kw", "rigid_kw", "flex_kw",
    "monthly_uc_energy_kwh", "uc_tariff", "der_tariff", "der_bid_cap", "dr_price_per_kwh",
    "dr_procurement_cap_kw", "event_active", "der_generation_kw", "grid_import_max_kw",
    "discharge_available_kw", "grid_import_offer_kw", "der_planned_curtailment_kw",
    "der_to_ess_kw", "grid_to_ess_kw", "ess_discharge_offer_kw", "der_capacity_kw",
    "uc_capacity_kw", "backlog_limit_kwh")

def uc_fields(h):
    return UC_CURRENT_FIELDS + tuple(f"aggregate_due_bucket_{k}_kwh" for k in range(h + 1)) + UC_HISTORY_FIELDS

def consumer_fields(h):
    return CONSUMER_CURRENT_FIELDS + tuple(f"due_bucket_{k}_kwh" for k in range(h + 1))

def queue_buckets(consumer, period, horizon):
    buckets = np.zeros(horizon + 1, dtype=np.float64)
    for task in consumer.tasks:
        offset = max(0, task.deadline_period - period)
        buckets[offset] += task.remaining_kwh
    return buckets

def remaining_steps(state, config):
    return config.system.horizon_steps - state.period_id if config.system.episode_mode == "finite" else -1

def _array(values):
    with np.errstate(over="ignore"):
        result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Observation is not representable as finite float32")
    return result

def build_uc_observation(state, profiles, config):
    index = min(state.period_id, len(profiles) - 1)
    base, rigid, flex = profiles.loads_kw[index], profiles.rigid_kw[index], profiles.flex_kw[index]
    buckets = np.asarray([queue_buckets(c, state.period_id, config.market.shift_deadline_steps)
                          for c in state.consumers])
    due = np.minimum(buckets[:, 0] / config.system.dt_hours, flex)
    weights = base + due
    total = float(weights.sum())
    rate = sum(uc_price(state.timestamp, c.monthly_uc_energy_kwh, config) * weights[i]
               for i, c in enumerate(state.consumers)) / total if total > 0 else 0.0
    active, _ = event_context(state.timestamp, config)
    values = [state.period_id, remaining_steps(state, config), state.ess_energy_kwh,
              profiles.der_kw[index], config.system.grid_import_max_kw, profiles.grid_cost_per_kwh[index],
              float(active), config.market.dr_incentive_per_kwh if active else 0,
              float(base.sum()), float(rigid.sum()), float(flex.sum()), float(due.sum()), rate]
    return _array(values + buckets.sum(axis=0).tolist() + list(state.last_totals))

def build_consumer_observation(state, consumer, base, rigid, flex, der, offer, config, index):
    a = offer.action
    tariff = uc_price(state.timestamp, consumer.monthly_uc_energy_kwh, config)
    values = [state.period_id, remaining_steps(state, config), base, rigid, flex,
              consumer.monthly_uc_energy_kwh, tariff,
              offer.der_tariff, min(individual(config.market.der_bid_cap, index), tariff), offer.dr_incentive,
              a.dr_procurement_cap_kw, float(offer.event_active), der, config.system.grid_import_max_kw,
              offer.discharge_available_kw, a.grid_import_offer_kw, a.der_planned_curtailment_kw,
              a.der_to_ess_kw, a.grid_to_ess_kw, a.ess_discharge_offer_kw,
              offer.der_capacity_kw, offer.uc_capacity_kw, individual(config.market.shift_backlog_limit_kwh, index)]
    return _array(values + queue_buckets(consumer, state.period_id, config.market.shift_deadline_steps).tolist())

