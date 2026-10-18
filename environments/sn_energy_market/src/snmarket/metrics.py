"""Aggregate committed v9 records without mixing cash, utility and physical capacity."""

import math


def summarize(records, dt_hours):
    records = tuple(records)
    if not math.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError("dt_hours must be finite and positive")
    if not records:
        return {"physical_periods": 0}

    def physical_energy(field):
        return math.fsum(getattr(r.physical, field) for r in records) * dt_hours

    def ledger_total(field):
        return math.fsum(getattr(r.ledger, field) for r in records)

    grid = [r.physical.grid_buy_kw for r in records]
    average = math.fsum(grid) / len(grid)
    generated = physical_energy("der_generation_kw")
    spill = physical_energy("der_spill_kw")
    consumer_ids = sorted({identity for r in records for identity in r.services})
    consumer_cost_fields = (
        "energy_payment", "der_payment", "uc_payment", "dr_income", "compensation_income", "net_bill",
        "adjustment_cost", "native_loss_cost", "waiting_cost", "overdue_cost",
        "terminal_pending_cost", "total_cost",
    )
    consumer_totals = {identity: {
        field: math.fsum(getattr(r.ledger.consumers[identity], field)
                         for r in records if identity in r.ledger.consumers)
        for field in consumer_cost_fields
    } for identity in consumer_ids}
    costs = {field: ledger_total(field) for field in (
        "energy_revenue", "dr_payment", "compensation_payment", "grid_cost",
        "der_operating_cost", "external_cost", "cash_profit", "ess_cost",
        "curtailment_cost", "book_profit", "supply_shortfall_cost", "due_cost",
        "uc_service_cost", "dr_value", "terminal_energy_value", "terminal_pending_cost", "uc_reward",
    )}
    return {
        "physical_periods": len(records),
        "consumer_service_records": sum(len(r.services) for r in records),
        "grid_energy_kwh": physical_energy("grid_buy_kw"),
        "grid_peak_kw": max(grid),
        "grid_par": max(grid) / average if average > 0 else None,
        "grid_ramp_sum_kw": math.fsum(abs(b - a) for a, b in zip(grid, grid[1:])),
        "der_generated_kwh": generated,
        "der_curtailed_kwh": spill,
        "der_planned_curtailment_kwh": physical_energy("der_planned_curtailment_kw"),
        "der_unsold_kwh": physical_energy("der_unsold_kw"),
        "der_utilization": (generated - spill) / generated if generated > 0 else None,
        "der_to_storage_kwh": physical_energy("der_to_ess_kw"),
        "grid_to_storage_kwh": physical_energy("grid_to_ess_kw"),
        "storage_discharge_kwh": physical_energy("ess_discharge_kw"),
        "ending_storage_kwh": records[-1].physical.ess_energy_after_kwh,
        "actual_service_kwh": physical_energy("supply_served_kw"),
        "der_sales_kwh": math.fsum(r.market.der_total_kw for r in records) * dt_hours,
        "uc_sales_kwh": math.fsum(r.market.uc_total_kw for r in records) * dt_hours,
        "native_lost_kwh": math.fsum(s.native_lost_kw for r in records for s in r.services.values()) * dt_hours,
        "actual_repayment_kwh": math.fsum(s.repay_actual_kw for r in records for s in r.services.values()) * dt_hours,
        "net_dr_kwh": math.fsum(r.dr.net_dr_kw for r in records) * dt_hours,
        "procured_dr_kwh": math.fsum(r.dr.procured_kw for r in records) * dt_hours,
        "supply_dr_credit_kwh": math.fsum(r.dr.supply_credit_kw for r in records) * dt_hours,
        "supply_reference_kwh": ledger_total("supply_reference_kw") * dt_hours,
        "announced_capacity_kwh": ledger_total("announced_capacity_kw") * dt_hours,
        "announced_shortfall_kwh": ledger_total("supply_shortfall_kw") * dt_hours,
        "unused_grid_offer_kwh": physical_energy("grid_offer_unused_kw"),
        "unused_discharge_offer_kwh": physical_energy("ess_discharge_unused_kw"),
        "ending_pending_kwh": math.fsum(s.backlog_after_kwh for s in records[-1].services.values()),
        "ending_due_kwh": math.fsum(s.overdue_after_kwh for s in records[-1].services.values()),
        "terminal_pending_kwh": math.fsum(s.backlog_after_kwh for r in records if r.terminated
                                         for s in r.services.values()),
        "uc_return": math.fsum(r.rewards["uc"] for r in records),
        "consumer_returns": {i: math.fsum(r.rewards[i] for r in records if i in r.rewards)
                             for i in consumer_ids},
        "uc_accounts": costs,
        "consumer_accounts": consumer_totals,
        "max_power_residual_kw": max(abs(r.physical.power_residual_kw) for r in records),
        "max_storage_energy_residual_kwh": max(abs(r.physical.energy_residual_kwh) for r in records),
        "max_queue_residual_kwh": max(abs(s.queue_residual_kwh) for r in records for s in r.services.values()),
        "max_cash_residual": max(abs(r.ledger.cash_residual) for r in records),
    }

