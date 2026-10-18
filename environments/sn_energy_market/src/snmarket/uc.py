"""Six physical UC decisions; validation publishes their frozen two-pool capacity."""

from .action import action_tuple, coerce_uc_action
from .config import VERSION
from .contracts import PublicOffer
from .tariff import event_context


def discharge_available(state, config):
    s = config.system
    return max(0.0, min(s.ess_discharge_max_kw,
                       s.ess_discharge_efficiency * (state.ess_energy_kwh - s.ess_energy_min_kwh)
                       / s.dt_hours))


def validate_uc_action(value, state, generation_kw, config):
    action = coerce_uc_action(value)
    if any(v < 0.0 for v in action_tuple(action)):
        raise ValueError("UC physical actions must be nonnegative")
    s = config.system
    tol = 1e-9
    if action.grid_import_offer_kw > s.grid_import_max_kw + tol:
        raise ValueError("PG offer exceeds the physical import limit")
    if action.grid_to_ess_kw > action.grid_import_offer_kw + tol:
        raise ValueError("PG charging exceeds its offer")
    if action.der_planned_curtailment_kw + action.der_to_ess_kw > generation_kw + tol:
        raise ValueError("DER curtailment and charging exceed generation")
    charge_limit = min(s.ess_charge_max_kw,
                       (s.ess_energy_capacity_kwh - state.ess_energy_kwh)
                       / (s.ess_charge_efficiency * s.dt_hours))
    if action.charge_kw > charge_limit + tol:
        raise ValueError("Charging exceeds power or SOC space")
    if action.ess_discharge_offer_kw > discharge_available(state, config) + tol:
        raise ValueError("Discharge offer exceeds power or stored energy")
    if action.charge_kw > 0.0 and action.ess_discharge_offer_kw > 0.0:
        raise ValueError("Charging and a positive discharge offer are mutually exclusive")
    active, _ = event_context(state.timestamp, config)
    if action.dr_procurement_cap_kw > (config.market.dr_procurement_max_kw if active else 0.0) + tol:
        raise ValueError("DR cap exceeds the event allowance")
    return action


def prepare_offer(value, state, generation_kw, config):
    action = validate_uc_action(value, state, generation_kw, config)
    active, event_id = event_context(state.timestamp, config)
    return PublicOffer(
        action=action,
        der_capacity_kw=max(0.0, generation_kw - action.der_planned_curtailment_kw - action.der_to_ess_kw),
        uc_capacity_kw=max(0.0, action.grid_import_offer_kw + action.ess_discharge_offer_kw
                           - action.grid_to_ess_kw),
        der_tariff=config.market.der_reference_rate_per_kwh,
        dr_incentive=config.market.dr_incentive_per_kwh if active else 0.0,
        event_active=active, event_id=event_id, period_id=state.period_id,
        rule_version=VERSION, discharge_available_kw=discharge_available(state, config),
    )

