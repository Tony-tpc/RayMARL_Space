"""v9 single-period cash transfers, utility costs, adequacy assessment and terminal values."""

import math

from .config import individual
from .contracts import ConsumerLedger, Ledger


def settle(orders, services, market, physical, dr, offer, state, grid_rate, terminated, config):
    orders = tuple(orders)
    r = config.rewards
    dt = config.system.dt_hours
    consumers = {}
    for order in orders:
        identity, index = order.consumer_id, order.consumer_index
        allocation, service = market.allocations[identity], services[identity]
        der_payment = order.action.der_bid_per_kwh * allocation.der_kw * dt
        uc_payment = order.uc_tariff * allocation.uc_kw * dt
        energy_payment = der_payment + uc_payment
        compensation = individual(r.actual_compensation_per_kwh, index) * service.native_lost_kw * dt
        dr_income = dr.consumers[identity].payment
        bill = energy_payment - dr_income - compensation
        adjustment = (individual(r.adjustment_rate_per_kwh, index)
                      * (order.action.curtail_kw ** 2 + max(order.action.signed_shift_kw, 0.0) ** 2)
                      / max(order.baseline_kw, r.adjustment_epsilon_kw) * dt)
        native = individual(r.consumer_native_loss_rate_per_kwh, index) * service.native_lost_kw * dt
        waiting = individual(r.consumer_wait_rate_per_kwh_hour, index) * service.backlog_after_kwh * dt
        overdue = individual(r.consumer_overdue_rate_per_kwh_hour, index) * service.overdue_after_kwh * dt
        terminal = (individual(r.consumer_terminal_pending_rate_per_kwh, index) * service.backlog_after_kwh
                    if terminated else 0.0)
        cost = bill + adjustment + native + waiting + overdue + terminal
        consumers[identity] = ConsumerLedger(
            energy_payment, der_payment, uc_payment, dr_income, compensation, bill,
            adjustment, native, waiting, overdue, terminal, cost,
        )
    energy_revenue = math.fsum(c.energy_payment for c in consumers.values())
    compensation_payment = math.fsum(c.compensation_income for c in consumers.values())
    grid_cost = grid_rate * physical.grid_buy_kw * dt
    der_cost = r.der_operating_rate_per_kwh * (physical.der_generation_kw - physical.der_spill_kw) * dt
    external = r.external_cost_per_period
    net_bills = math.fsum(c.net_bill for c in consumers.values())
    cash = net_bills - grid_cost - der_cost - external
    ess = r.ess_throughput_rate_per_kwh * (physical.ess_charge_kw + physical.ess_discharge_kw) * dt
    curtail = r.curtailment_rate_per_kwh * physical.der_spill_kw * dt
    book = cash - ess - curtail
    reference = math.fsum(o.baseline_kw + o.due_kw for o in orders)
    announced = offer.der_capacity_kw + offer.uc_capacity_kw
    gap = max(0.0, reference - announced - dr.supply_credit_kw)
    shortfall_cost = r.uc_supply_shortfall_rate_per_kwh * gap * dt
    due_cost = r.uc_due_rate_per_kwh * math.fsum(s.overdue_after_kwh for s in services.values())
    uc_service = shortfall_cost + due_cost
    value = r.dr_value_rate_per_kwh * dr.net_dr_kw * dt
    terminal_energy = (r.terminal_energy_value_per_kwh
                       * (physical.ess_energy_after_kwh - state.initial_ess_energy_kwh)
                       if terminated else 0.0)
    terminal_pending = (r.uc_terminal_pending_rate_per_kwh
                        * math.fsum(s.backlog_after_kwh for s in services.values()) if terminated else 0.0)
    reward = cash - ess - curtail + value - uc_service + terminal_energy - terminal_pending
    cash_residual = cash + grid_cost + der_cost + external - net_bills
    return Ledger(
        consumers, energy_revenue, dr.total_payment, compensation_payment, grid_cost, der_cost, external,
        cash, ess, curtail, book, reference, announced, dr.supply_credit_kw, gap, shortfall_cost,
        due_cost, uc_service, value, terminal_energy, terminal_pending, reward, cash_residual,
    )

