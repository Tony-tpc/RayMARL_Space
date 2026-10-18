"""Production/oracle bridge: extraction and exhaustive comparison, no formulas."""
import math
from dataclasses import astuple

import numpy as np
from v9_oracle import evaluate

RATE_FIELDS = {
    "der_operation": "der_operating_rate_per_kwh",
    "throughput": "ess_throughput_rate_per_kwh", "spill": "curtailment_rate_per_kwh",
    "dr_value": "dr_value_rate_per_kwh", "adequacy": "uc_supply_shortfall_rate_per_kwh",
    "uc_due": "uc_due_rate_per_kwh", "compensation": "actual_compensation_per_kwh",
    "loss": "consumer_native_loss_rate_per_kwh", "discomfort": "adjustment_rate_per_kwh",
    "waiting": "consumer_wait_rate_per_kwh_hour", "late": "consumer_overdue_rate_per_kwh_hour",
    "uc_terminal": "uc_terminal_pending_rate_per_kwh", "terminal": "consumer_terminal_pending_rate_per_kwh",
    "terminal_energy": "terminal_energy_value_per_kwh", "epsilon": "adjustment_epsilon_kw",
    "external": "external_cost_per_period",
}


def oracle_for(engine, state, uc_action, consumer_actions):
    """Derive tariff/event independently from configured tables and timestamp."""
    config, profiles = engine.config, engine.profiles
    sys, market = config.system, config.market
    t = state.period_id
    day_step = (state.timestamp.hour * 60 + state.timestamp.minute) // sys.resolution_minutes
    event = any(start <= day_step < end for start, end in market.dr_event_intervals)
    hour = state.timestamp.hour + state.timestamp.minute / 60
    peak = any(start <= hour < stop for start, stop in market.uc_peak_intervals_hours)
    high = any(start <= hour < stop for start, stop in market.uc_high_intervals_hours)
    rate_table = market.uc_peak_rates if peak else market.uc_high_rates if high else market.uc_flat_rates
    people = []
    for i, consumer in enumerate(state.consumers):
        action = consumer_actions[consumer.consumer_id]
        action = astuple(action) if hasattr(action, "__dataclass_fields__") else tuple(action)
        tier = sum(consumer.monthly_uc_energy_kwh >= threshold for threshold in market.tier_thresholds_kwh)
        backlog = market.shift_backlog_limit_kwh
        people.append(dict(base=float(profiles.loads_kw[t, i]), rigid=float(profiles.rigid_kw[t, i]),
                           flex=float(profiles.flex_kw[t, i]),
                           backlog_limit=float(backlog[i] if isinstance(backlog, tuple) else backlog),
                           tariff=float(rate_table[tier]), cut=action[0], shift=action[1], rho=action[2],
                           bid=action[3], M=consumer.monthly_uc_energy_kwh,
                           tasks=tuple((task.task_id, task.source_period, task.deadline_period, task.remaining_kwh)
                                       for task in consumer.tasks)))
    uc = astuple(uc_action) if hasattr(uc_action, "__dataclass_fields__") else tuple(uc_action)
    return evaluate(people, uc, der=float(profiles.der_kw[t]), grid_limit=sys.grid_import_max_kw,
                    energy=state.ess_energy_kwh, energy_min=sys.ess_energy_min_kwh,
                    energy_max=sys.ess_energy_capacity_kwh, charge_max=sys.ess_charge_max_kw,
                    discharge_max=sys.ess_discharge_max_kw, eta_c=sys.ess_charge_efficiency,
                    eta_d=sys.ess_discharge_efficiency, dt=sys.dt_hours, period=t,
                    deadline=market.shift_deadline_steps,
                    horizon=sys.horizon_steps if sys.episode_mode == "finite" else None,
                    event=event, dr_price=market.dr_incentive_per_kwh if event else 0.0,
                    dr_limit=market.dr_procurement_max_kw, grid_price=float(profiles.grid_cost_per_kwh[t]),
                    rates={key: getattr(config.rewards, field) for key, field in RATE_FIELDS.items()},
                    terminal=sys.episode_mode == "finite" and t + 1 == sys.horizon_steps,
                    initial_energy=state.initial_ess_energy_kwh)


def compare_record(engine, before, record, *, atol=1e-8):
    """Return maximum residual; raise with field names on any disagreement."""
    expected = oracle_for(engine, before, record.uc_action, record.consumer_actions)
    actual, desired, labels = [], [], []

    def add(label, value, oracle):
        labels.append(label)
        actual.append(value)
        desired.append(oracle)

    for field, key in (("grid_buy_kw", "grid"), ("ess_discharge_kw", "discharge"),
                       ("der_spill_kw", "spill"), ("der_unsold_kw", "unsold"),
                       ("ess_energy_after_kwh", "energy_after")):
        add(field, getattr(record.physical, field), expected[key])
    for field, key in (("der_capacity_kw", "der_capacity"), ("uc_capacity_kw", "uc_capacity")):
        add(field, getattr(record.offer, field), expected[key])
    for field, key in (("positive_dr_kw", "gross_dr"), ("net_dr_kw", "net_dr"),
                       ("procured_kw", "paid_dr"), ("supply_credit_kw", "credit_dr")):
        add(field, getattr(record.dr, field), expected[key])
    for field, key in (("cash_profit", "cash"), ("grid_cost", "grid_cash"),
                       ("der_operating_cost", "der_cash"), ("ess_cost", "throughput_cost"),
                       ("curtailment_cost", "spill_cost"), ("book_profit", "book"),
                       ("supply_reference_kw", "reference"), ("supply_shortfall_kw", "adequacy_gap"),
                       ("supply_shortfall_cost", "adequacy_cost"), ("due_cost", "due_cost"),
                       ("dr_value", "dr_value"), ("uc_reward", "reward")):
        add(field, getattr(record.ledger, field), expected[key])
    add("terminal", record.ledger.terminal_energy_value - record.ledger.terminal_pending_cost,
        expected["terminal"])
    add("UC reward", record.rewards["uc"], expected["reward"])
    for i, order in enumerate(record.orders):
        identity = order.consumer_id
        want = expected["people"][i]
        for field, key in (("native_kw", "native"), ("request_kw", "request"), ("necessary_kw", "necessary")):
            add(identity + field, getattr(order, field), want[key])
        allocation, service = record.market.allocations[identity], record.services[identity]
        add(identity + "DER", allocation.der_kw, want["der"])
        add(identity + "UC", allocation.uc_kw, want["uc"])
        for layer, name in enumerate(("necessary", "early")):
            for field, key in (("der_first_kw", "der_initial"), ("uc_kw", "uc"),
                               ("der_rescue_kw", "der_rescue"), ("unfilled_kw", "gap")):
                add(identity + name + field, getattr(getattr(allocation, name), field),
                    expected["layers"][layer][key][i])
        for field, key in (("native_lost_kw", "lost"), ("repay_actual_kw", "repay"),
                           ("backlog_after_kwh", "backlog"), ("overdue_after_kwh", "due_after")):
            add(identity + field, getattr(service, field), want[key])
        got_tasks = [(t.source_period, t.deadline_period, t.remaining_kwh) for t in service.tasks_after]
        want_tasks = [(t[1], t[2], t[3]) for t in want["tasks"]]
        # A completed subtraction can leave an oracle-only ulp-sized queue tail.
        # Compare energy for every provenance key rather than interpreting its
        # presence as a new physical task. Different FIFO repayment choices
        # still disagree at their creation/deadline keys. Dedicated regressions
        # separately require genuine 1e-13 kWh tasks to retain their topology.
        keys = {(t[0], t[1]) for t in got_tasks + want_tasks}
        for key in sorted(keys):
            got = math.fsum(t[2] for t in got_tasks if t[:2] == key)
            wanted = math.fsum(t[2] for t in want_tasks if t[:2] == key)
            add(identity + f" task energy {key}", got, wanted)
        ledger = record.ledger.consumers[identity]
        for field, key in (("energy_payment", "energy_cash"), ("dr_income", "dr_cash"),
                           ("compensation_income", "loss_cash"), ("net_bill", "bill"),
                           ("adjustment_cost", "flex_cost")):
            add(identity + field, getattr(ledger, field), want[key])
        add(identity + "service cost", ledger.native_loss_cost + ledger.waiting_cost + ledger.overdue_cost,
            want["service_cost"])
        add(identity + "terminal cost", -ledger.terminal_pending_cost, want["terminal"])
        add(identity + "reward", record.rewards[identity], want["reward"])
    delta = np.abs(np.asarray(actual) - np.asarray(desired))
    if not np.allclose(actual, desired, atol=atol, rtol=1e-9):
        bad = [(labels[i], actual[i], desired[i]) for i in np.flatnonzero(delta > atol)]
        raise AssertionError(bad[:20])
    return {"maximum_absolute_residual": float(np.max(delta, initial=0.0)), "compared_fields": len(actual)}
