"""Execute firm charging/curtailment and use only the previously announced offers."""

import math

from .contracts import PhysicalExecution


def execute(offer, market, generation_kw, state, config):
    a, s = offer.action, config.system
    if market.der_total_kw > offer.der_capacity_kw + 1e-8 or market.uc_total_kw > offer.uc_capacity_kw + 1e-8:
        raise ValueError("Sales exceed announced supply")
    discharge = min(a.ess_discharge_offer_kw, market.uc_total_kw)
    grid = market.uc_total_kw + a.grid_to_ess_kw - discharge
    unsold = max(0.0, offer.der_capacity_kw - market.der_total_kw)
    spill = a.der_planned_curtailment_kw + unsold
    charge = a.charge_kw
    energy = (state.ess_energy_kwh + s.ess_charge_efficiency * charge * s.dt_hours
              - discharge * s.dt_hours / s.ess_discharge_efficiency)
    served = market.der_total_kw + market.uc_total_kw
    residual = grid + generation_kw + discharge - charge - served - spill
    energy_tol = 1e-8 + 1e-10 * max(1.0, abs(energy), s.ess_energy_capacity_kwh)
    if (not all(math.isfinite(v) for v in (grid, energy, served, residual))
            or grid < -1e-8 or grid > a.grid_import_offer_kw + 1e-8
            or energy < s.ess_energy_min_kwh - energy_tol
            or energy > s.ess_energy_capacity_kwh + energy_tol
            or abs(residual) > 1e-8 * max(1.0, grid + generation_kw + discharge)):
        error = ValueError("Physical power/SOC accounting failed")
        error.residuals = {"power_balance_kw": residual, "grid_import_kw": grid, "SOC_kwh": energy}
        raise error
    # Only normalize round-off beyond the storage interval, after checking the raw result.
    # All submitted actions and physical power flows remain untouched.
    raw_energy = energy
    energy = min(s.ess_energy_capacity_kwh, max(s.ess_energy_min_kwh, energy))
    return PhysicalExecution(
        grid, generation_kw, a.der_planned_curtailment_kw, unsold, spill,
        a.der_to_ess_kw, a.grid_to_ess_kw, discharge,
        max(0.0, a.grid_import_offer_kw - grid), max(0.0, a.ess_discharge_offer_kw - discharge),
        energy, served, residual, energy - raw_energy,
    )

