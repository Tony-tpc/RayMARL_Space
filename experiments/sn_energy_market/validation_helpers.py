"""Independent scenario/action generation and non-RL v9 production checks.

The samplers below use physical constraints directly. They intentionally do not
call the policy output mapping or production local-bound functions, so invalid
constraint implementations cannot be hidden by a matching action decoder.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "environments/sn_energy_market/src"))


def json_safe(value):
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def personal(value, index):
    return float(value[index] if isinstance(value, (tuple, list)) else value)


def near(actual, expected, label, maximums=None, unit="numeric"):
    residual = abs(float(actual) - float(expected))
    if maximums is not None:
        maximums[f"{unit}_residual"] = max(maximums.get(f"{unit}_residual", 0.0), residual)
    tolerance = 1e-8 + 1e-10 * max(abs(float(actual)), abs(float(expected)))
    if not math.isfinite(residual) or residual > tolerance:
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}, residual={residual!r}")


def fraction(rng, index):
    return 0.0 if index % 7 == 0 else 1.0 if index % 7 == 1 else float(rng.random())


def event_at(timestamp, config):
    step = (timestamp.hour * 60 + timestamp.minute) // config.system.resolution_minutes
    return any(start <= step < stop for start, stop in config.market.dr_event_intervals)


def tariff_band(timestamp, config):
    hour = timestamp.hour + timestamp.minute / 60
    if any(start <= hour < stop for start, stop in config.market.uc_peak_intervals_hours):
        return "peak"
    if any(start <= hour < stop for start, stop in config.market.uc_high_intervals_hours):
        return "high"
    return "flat"


def tariff_at(timestamp, monthly, config):
    level = sum(monthly >= threshold for threshold in config.market.tier_thresholds_kwh)
    prices = getattr(config.market, f"uc_{tariff_band(timestamp, config)}_rates")
    return float(prices[level])


def scenario(master_seed, index, count, periods, kind="episode"):
    from snmarket import (
        ConsumerState,
        EnergyMarketEngine,
        EnvironmentConfig,
        Profiles,
        ShiftTask,
    )

    kind_number = {"static": 1, "invalid": 2, "episode": 3, "long": 4, "month": 5, "smoke": 6}[kind]
    sequence = np.random.SeedSequence([master_seed, kind_number, index])
    profile_stream, action_stream = sequence.spawn(2)
    rng = np.random.default_rng(profile_stream)
    continuous = kind in {"static", "invalid", "long", "month"} or index % 2 == 1
    minutes = 15 if kind in {"long", "month", "smoke"} else (5, 15, 30, 60)[index % 4]
    delay = (1, 4, 24, 96)[index % 4] if kind not in {"long", "month"} else 96
    capacity = max(5.0, count * float(rng.uniform(1.0, 4.0)))
    initial_energy = capacity * (0.0, 1.0, 0.25, 0.75)[index % 4]
    start = "2026-01-31T23:00:00+08:00" if index % 3 == 0 or kind == "month" else "2026-02-10T00:00:00+08:00"
    base = EnvironmentConfig()
    system = dataclasses.replace(base.system, num_consumers=count, horizon_steps=periods,
        episode_mode="continuous" if continuous else "finite", resolution_minutes=minutes,
        grid_import_max_kw=count * float(rng.uniform(1.0, 15.0)),
        ess_energy_capacity_kwh=capacity, ess_energy_min_kwh=0.0,
        ess_initial_energy_kwh=initial_energy, ess_charge_max_kw=count * float(rng.uniform(0.5, 6.0)),
        ess_discharge_max_kw=count * float(rng.uniform(0.5, 6.0)),
        ess_charge_efficiency=float(rng.uniform(0.75, 1.0)),
        ess_discharge_efficiency=float(rng.uniform(0.75, 1.0)), start_datetime=start)
    day = system.steps_per_day
    events = () if index % 7 == 0 else ((0, day),) if index % 7 == 1 else ((day // 3, 2 * day // 3),)
    schedules = (
        (((9.0, 15.0),), ((20.0, 22.0),)),
        ((), ()),
        (((0.0, 6.0), (12.0, 18.0)), ((6.0, 9.0), (21.0, 24.0))),
        (((1.25, 3.75), (14.5, 16.5)), ((8.5, 9.25), (20.25, 22.5))),
    )
    high_intervals, peak_intervals = schedules[(index // 4) % len(schedules)]
    limits = tuple(float(x) for x in rng.uniform(1.0, 8.0, count))
    market = dataclasses.replace(base.market, shift_deadline_steps=delay,
        shift_backlog_limit_kwh=limits, der_bid_cap=tuple(float(x) for x in rng.uniform(0.3, 1.2, count)),
        dr_event_intervals=events, dr_procurement_max_kw=count * float(rng.uniform(0.0, 6.0)),
        uc_high_intervals_hours=high_intervals, uc_peak_intervals_hours=peak_intervals,
        dr_incentive_per_kwh=0.0 if index % 11 == 0 else float(rng.uniform(0.05, 1.0)),
        tier_thresholds_kwh=(1.0, 4.0))
    coefficients = lambda low, high: tuple(float(x) for x in rng.uniform(low, high, count))
    rewards = dataclasses.replace(base.rewards,
        consumer_native_loss_rate_per_kwh=coefficients(1.0, 4.0),
        consumer_wait_rate_per_kwh_hour=coefficients(0.0, 0.3),
        consumer_overdue_rate_per_kwh_hour=coefficients(0.0, 1.0),
        consumer_terminal_pending_rate_per_kwh=coefficients(0.0, 4.0),
        actual_compensation_per_kwh=coefficients(0.0, 2.0),
        adjustment_rate_per_kwh=coefficients(0.05, 1.0),
        uc_supply_shortfall_rate_per_kwh=float(rng.uniform(0.0, 3.0)),
        uc_due_rate_per_kwh=float(rng.uniform(0.0, 2.0)),
        dr_value_rate_per_kwh=0.0 if index % 2 else float(rng.uniform(0.0, 0.4)),
        der_operating_rate_per_kwh=float(rng.uniform(0.0, 0.15)),
        terminal_energy_value_per_kwh=float(rng.uniform(0.0, 0.6)),
        external_cost_per_period=float(rng.uniform(0.0, 0.1)))
    config = dataclasses.replace(base, system=system, market=market, rewards=rewards).validate()
    time_index = np.arange(periods)[:, None]
    daily = 0.5 + 0.5 * np.sin(time_index * 2 * np.pi / day + rng.uniform(0, 2 * np.pi, (1, count)))
    loads = rng.uniform(0.1, 8.0, (1, count)) * (0.3 + daily) + rng.uniform(0, 2, (periods, count))
    rigid = loads * rng.uniform(0.0, 0.8, (periods, count))
    flex = rng.uniform(0.0, 10.0, (periods, count))
    der = rng.uniform(0, count * 10.0, periods)
    prices = rng.uniform(0.05, 1.7, periods)
    if index % 13 == 0:
        der[::3] = 0.0
    if index % 17 == 0:
        loads[::5] = 0.0
        rigid[::5] = 0.0
    if index % 19 == 0:
        flex[::4] = 0.0
    profiles = Profiles(loads, der, prices, flex, rigid_kw=rigid,
                        start_datetime=start, resolution_minutes=minutes,
                        metadata={"source": f"validation_{kind}", "value_kind": "realized",
                                  "seed": [master_seed, kind_number, index], "split": "validation"})
    consumers = []
    for i in range(count):
        tasks = []
        if (index + i) % 5:
            deadlines = sorted({-1, 0, max(0, delay // 2), delay - 1})
            amounts = rng.dirichlet(np.ones(len(deadlines))) * limits[i] * fraction(rng, index + i)
            tasks = [ShiftTask(d - delay, d, float(e),
                     task_id=f"consumer_{i}:seed:{j}" if (index + i + j) % 2 else None)
                     for j, (d, e) in enumerate(zip(deadlines, amounts)) if e > 0]
        consumers.append(ConsumerState(f"consumer_{i}", float(rng.choice([0, 1, 4, 4.5])), tuple(tasks)))
    engine = EnergyMarketEngine(config, profiles)
    engine.reset(seed=master_seed + index, options={"consumer_states": tuple(consumers),
                 "episode_id": f"validation-{kind}-{master_seed}-{index}"})
    return engine, np.random.default_rng(action_stream)


def sample_uc(state, der, config, rng, index, fixed=False):
    from snmarket import UCAction

    system = config.system
    available_discharge = min(system.ess_discharge_max_kw,
        system.ess_discharge_efficiency * (state.ess_energy_kwh - system.ess_energy_min_kwh) / system.dt_hours)
    if fixed or index % 11 == 0:
        return UCAction(grid_import_offer_kw=system.grid_import_max_kw, der_planned_curtailment_kw=0.0, der_to_ess_kw=0.0, grid_to_ess_kw=0.0,
                        ess_discharge_offer_kw=max(0.0, available_discharge),
                        dr_procurement_cap_kw=config.market.dr_procurement_max_kw if event_at(state.timestamp, config) else 0.0)
    grid = system.grid_import_max_kw * fraction(rng, index)
    curtail = der * fraction(rng, index + 3)
    charge_der = charge_pg = discharge = 0.0
    if index % 3 == 0:
        room = min(system.ess_charge_max_kw,
            (system.ess_energy_capacity_kwh - state.ess_energy_kwh)
            / (system.ess_charge_efficiency * system.dt_hours), der - curtail + grid)
        total = max(0.0, room) * fraction(rng, index + 1)
        lower, upper = max(0.0, total - grid), min(total, der - curtail)
        charge_der = lower + max(0.0, upper - lower) * fraction(rng, index + 2)
        charge_pg = total - charge_der
    elif index % 3 == 1:
        discharge = max(0.0, available_discharge) * fraction(rng, index + 4)
    procurement = config.market.dr_procurement_max_kw * fraction(rng, index + 5)
    if not event_at(state.timestamp, config):
        procurement = 0.0
    return UCAction(grid_import_offer_kw=grid, der_planned_curtailment_kw=curtail, der_to_ess_kw=charge_der, grid_to_ess_kw=charge_pg, ess_discharge_offer_kw=discharge, dr_procurement_cap_kw=procurement)


def sample_consumers(state, profiles, config, rng, index, fixed=False):
    from snmarket import ConsumerAction

    actions = {}
    dt, period = config.system.dt_hours, state.period_id
    for i, consumer in enumerate(state.consumers):
        base, rigid, flex = profiles.loads_kw[period, i], profiles.rigid_kw[period, i], profiles.flex_kw[period, i]
        backlog = math.fsum(task.remaining_kwh for task in consumer.tasks)
        due = min(math.fsum(task.remaining_kwh for task in consumer.tasks if task.deadline_period <= period) / dt, flex)
        cut_max = min(base - rigid, flex - due)
        cut = max(0.0, cut_max) * (0.25 if fixed else fraction(rng, index + i))
        lower = -min(flex - cut, backlog / dt)
        if due > 0:
            upper = -due
        else:
            upper = min(base - rigid - cut, flex - cut,
                        (personal(config.market.shift_backlog_limit_kwh, i) - backlog) / dt)
            if config.system.episode_mode == "finite" and period + config.market.shift_deadline_steps > config.system.horizon_steps - 1:
                upper = min(0.0, upper)
        assert lower <= upper + 1e-8
        shift = lower + max(0.0, upper - lower) * (0.25 if fixed else fraction(rng, index + i + 2))
        tariff = tariff_at(state.timestamp, consumer.monthly_uc_energy_kwh, config)
        bid_low = min(config.market.der_reference_rate_per_kwh, tariff)
        bid_high = min(personal(config.market.der_bid_cap, i), tariff)
        # Every fifth period creates exact ties, without rounding arbitrary bids.
        bid = bid_low if fixed or index % 5 == 0 else bid_low + (bid_high - bid_low) * fraction(rng, index + i + 4)
        bid = min(bid_high, max(bid_low, bid))
        rho = 0.5 if fixed else fraction(rng, index + i + 6)
        actions[consumer.consumer_id] = ConsumerAction(curtail_kw=cut, signed_shift_kw=shift, der_preference_fraction=rho, der_bid_per_kwh=bid)
    return actions


def check_observations(engine, before, uc_observation, consumer_observations, actions, maximums):
    config = engine.config
    horizon, dt = config.market.shift_deadline_steps, config.system.dt_hours
    if uc_observation.shape != (horizon + 20,) or uc_observation.dtype != np.float32:
        raise AssertionError("UC observation shape/dtype violates H+20 float32 contract")
    if not np.isfinite(uc_observation).all():
        raise AssertionError("Non-finite UC observation")
    if set(consumer_observations) != set(actions):
        raise AssertionError("Consumer actor set differs")
    aggregated = np.zeros(horizon + 1)
    for i, consumer in enumerate(before.consumers):
        obs = consumer_observations[consumer.consumer_id]
        if obs.shape != (horizon + 24,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise AssertionError("Consumer observation shape/dtype/finiteness")
        buckets = np.zeros(horizon + 1)
        for task in consumer.tasks:
            bucket = max(0, task.deadline_period - before.period_id)
            assert bucket <= horizon
            buckets[bucket] += task.remaining_kwh
        if not np.allclose(obs[23:], buckets, atol=2e-6, rtol=2e-6):
            raise AssertionError("Consumer deadline buckets differ from private FIFO")
        aggregated += buckets
        expected_remaining = -1 if config.system.episode_mode == "continuous" else config.system.horizon_steps - before.period_id
        if obs[0] != before.period_id or obs[1] != expected_remaining:
            raise AssertionError("Consumer time/horizon convention")
        if not np.isclose(obs[2], engine.profiles.loads_kw[before.period_id, i], atol=2e-6, rtol=2e-6):
            raise AssertionError("Consumer observed baseline differs")
    # float32 observation conversion has a separate tolerance from the float64 engine.
    if not np.allclose(uc_observation[13:14 + horizon], aggregated, atol=2e-5, rtol=2e-6):
        raise AssertionError("UC aggregated deadline buckets differ")
    due = math.fsum(min(math.fsum(t.remaining_kwh for t in c.tasks if t.deadline_period <= before.period_id) / dt,
                       engine.profiles.flex_kw[before.period_id, i]) for i, c in enumerate(before.consumers))
    if not math.isclose(float(uc_observation[11]), due, abs_tol=2e-5, rel_tol=2e-6):
        raise AssertionError("UC due is not sum of individually capacity-limited obligations")


def check_record(engine, before, uc, actions, record, coverage, maximums):
    """Independent arithmetic identities for every physical period, not selected samples."""
    config, p, ledger = engine.config, record.physical, record.ledger
    system, market, rewards, dt = config.system, config.market, config.rewards, config.system.dt_hours
    assert record.uc_action == uc and record.consumer_actions == actions
    assert record.period_id == before.period_id and record.offer.action == uc
    near(p.der_to_ess_kw, uc.der_to_ess_kw, "firm DER charge")
    near(p.grid_to_ess_kw, uc.grid_to_ess_kw, "firm PG charge")
    near(p.der_planned_curtailment_kw, uc.der_planned_curtailment_kw, "firm curtailment")
    assert -1e-8 <= p.grid_buy_kw <= uc.grid_import_offer_kw + 1e-8
    assert -1e-8 <= p.ess_discharge_kw <= uc.ess_discharge_offer_kw + 1e-8
    near(p.ess_discharge_kw, min(uc.ess_discharge_offer_kw, record.market.uc_total_kw), "ESS execution")
    near(p.grid_buy_kw, record.market.uc_total_kw + uc.grid_to_ess_kw - p.ess_discharge_kw, "PG execution")
    near(p.ess_energy_after_kwh, before.ess_energy_kwh + system.ess_charge_efficiency * uc.charge_kw * dt
         - p.ess_discharge_kw * dt / system.ess_discharge_efficiency, "SOC", maximums, "energy")
    assert system.ess_energy_min_kwh - 1e-8 <= p.ess_energy_after_kwh <= system.ess_energy_capacity_kwh + 1e-8
    near(p.der_spill_kw, uc.der_planned_curtailment_kw + p.der_unsold_kw, "spill decomposition")
    near(p.grid_buy_kw + p.der_generation_kw + p.ess_discharge_kw,
         p.supply_served_kw + uc.charge_kw + p.der_spill_kw, "power balance", maximums, "power")
    request_total = math.fsum(order.request_kw for order in record.orders)
    near(math.fsum(a.served_kw for a in record.market.allocations.values()), p.supply_served_kw,
         "individual/aggregate service", maximums, "power")
    near(p.supply_served_kw, min(request_total, record.offer.der_capacity_kw + record.offer.uc_capacity_kw),
         "announced capacity utilization", maximums, "power")
    if math.fsum(a.necessary.unfilled_kw for a in record.market.allocations.values()) > 1e-8:
        near(math.fsum(a.early.der_kw + a.early.uc_kw for a in record.market.allocations.values()), 0, "necessary priority")
        coverage["necessary_shortage_periods"] += 1
    native_total = repay_total = total_bill = due_after_total = 0.0
    for i, (consumer, order) in enumerate(zip(before.consumers, record.orders)):
        name = consumer.consumer_id
        allocation, service, account, action = record.market.allocations[name], record.services[name], ledger.consumers[name], actions[name]
        assert order.consumer_id == name and allocation.served_kw <= order.request_kw + 1e-8
        near(order.uc_tariff, tariff_at(before.timestamp, consumer.monthly_uc_energy_kwh, config), "calendar tariff")
        previous_ids = [task.task_id for task in consumer.tasks]
        current_ids = [task.task_id for task in service.tasks_after]
        assert all(isinstance(value, str) and value for value in previous_ids + current_ids), "Missing stable task ID"
        assert len(previous_ids) == len(set(previous_ids)) and len(current_ids) == len(set(current_ids)), "Duplicate task ID"
        native = order.baseline_kw - action.curtail_kw - max(0.0, action.signed_shift_kw)
        request = order.baseline_kw - action.curtail_kw - action.signed_shift_kw
        near(order.native_kw, native, "native request")
        near(order.request_kw, request, "total request")
        near(service.native_lost_kw, max(0.0, native - allocation.served_kw), "native loss")
        near(service.repay_actual_kw, max(0.0, allocation.served_kw - native), "actual repayment")
        old_due = math.fsum(t.remaining_kwh for t in consumer.tasks if t.deadline_period <= before.period_id)
        near(service.overdue_after_kwh, max(0.0, old_due - service.repay_actual_kw * dt), "remaining due", maximums, "queue")
        near(service.backlog_after_kwh, consumer.backlog_kwh + (max(0.0, action.signed_shift_kw) - service.repay_actual_kw) * dt,
             "queue energy", maximums, "queue")
        assert -1e-8 <= service.backlog_after_kwh <= personal(market.shift_backlog_limit_kwh, i) + 1e-8
        left, expected_tasks = service.repay_actual_kw * dt, []
        for task in sorted(consumer.tasks, key=lambda task: (task.deadline_period, task.source_period)):
            remaining = max(0.0, task.remaining_kwh - left)
            left = max(0.0, left - task.remaining_kwh)
            if remaining > 1e-10:
                expected_tasks.append((task.source_period, task.deadline_period, remaining, task.task_id))
        if action.signed_shift_kw > 1e-10:
            expected_tasks.append((before.period_id, before.period_id + market.shift_deadline_steps, action.signed_shift_kw * dt, None))
        actual_tasks = [(t.source_period, t.deadline_period, t.remaining_kwh, t.task_id)
                        for t in service.tasks_after if t.remaining_kwh > 1e-10]
        assert len(expected_tasks) == len(actual_tasks), (expected_tasks, actual_tasks)
        for expected, actual in zip(expected_tasks, actual_tasks):
            assert expected[:2] == actual[:2]
            near(actual[2], expected[2], "FIFO task energy", maximums, "queue")
            if expected[3] is not None:
                assert actual[3] == expected[3], "Repayment changed stable task ID"
                coverage["retained_task_ids"] += 1
            else:
                coverage["new_task_ids"] += 1
        near(account.energy_payment, (action.der_bid_per_kwh * allocation.der_kw + order.uc_tariff * allocation.uc_kw) * dt,
             "energy payment", maximums, "cash")
        near(account.net_bill, account.energy_payment - account.dr_income - account.compensation_income, "bill", maximums, "cash")
        near(account.compensation_income, personal(rewards.actual_compensation_per_kwh, i) * service.native_lost_kw * dt,
             "cash loss compensation", maximums, "cash")
        near(record.rewards[name], -account.total_cost, "consumer reward", maximums, "reward")
        near(account.adjustment_cost, personal(rewards.adjustment_rate_per_kwh, i)
             * (action.curtail_kw ** 2 + max(0.0, action.signed_shift_kw) ** 2)
             / max(order.baseline_kw, rewards.adjustment_epsilon_kw) * dt, "discomfort", maximums, "cash")
        near(account.native_loss_cost, personal(rewards.consumer_native_loss_rate_per_kwh, i)
             * service.native_lost_kw * dt, "native loss cost")
        near(account.waiting_cost, personal(rewards.consumer_wait_rate_per_kwh_hour, i) * service.backlog_after_kwh * dt, "wait cost")
        near(account.overdue_cost, personal(rewards.consumer_overdue_rate_per_kwh_hour, i) * service.overdue_after_kwh * dt, "late cost")
        near(account.terminal_pending_cost, personal(rewards.consumer_terminal_pending_rate_per_kwh, i) * service.backlog_after_kwh if record.terminated else 0.0, "terminal once")
        near(account.total_cost, account.net_bill + account.adjustment_cost + account.native_loss_cost
             + account.waiting_cost + account.overdue_cost + account.terminal_pending_cost,
             "consumer full cost", maximums, "cash")
        native_total += service.native_lost_kw
        repay_total += service.repay_actual_kw
        total_bill += account.net_bill
        due_after_total += service.overdue_after_kwh
        coverage["consumer_records"] += 1
        coverage["due_consumer_periods"] += int(old_due > 0)
        coverage["due_over_capacity"] += int(old_due > order.flex_kw * dt + 1e-8)
        coverage["early_repayment"] += int(allocation.early.der_kw + allocation.early.uc_kw > 1e-8)
        coverage["shift_out"] += int(action.signed_shift_kw > 1e-8)
    signed = [float(record.offer.event_active) * (o.baseline_kw - o.request_kw) for o in record.orders]
    net = max(0.0, math.fsum(signed))
    near(record.dr.net_dr_kw, net, "net request DR")
    near(record.dr.procured_kw, min(uc.dr_procurement_cap_kw, net), "procured DR")
    near(record.dr.total_payment, record.offer.dr_incentive * record.dr.procured_kw * dt, "DR payment", maximums, "cash")
    near(math.fsum(v.payment for v in record.dr.consumers.values()), record.dr.total_payment, "DR split", maximums, "cash")
    credit = record.dr.procured_kw if record.offer.event_active and record.offer.dr_incentive > 0 else 0.0
    near(record.dr.supply_credit_kw, credit, "positive-price-event credit")
    reference = math.fsum(o.baseline_kw + o.due_kw for o in record.orders)
    gap = max(0.0, reference - record.offer.der_capacity_kw - record.offer.uc_capacity_kw - credit)
    near(ledger.supply_reference_kw, reference, "frozen supply reference")
    near(ledger.supply_shortfall_kw, gap, "announced supply gap")
    near(ledger.supply_shortfall_cost, rewards.uc_supply_shortfall_rate_per_kwh * gap * dt, "adequacy cost", maximums, "cash")
    near(ledger.due_cost, rewards.uc_due_rate_per_kwh * due_after_total, "UC due no extra dt", maximums, "cash")
    near(ledger.grid_cost, engine.profiles.grid_cost_per_kwh[before.period_id] * p.grid_buy_kw * dt,
         "external grid cost", maximums, "cash")
    near(ledger.der_operating_cost, rewards.der_operating_rate_per_kwh * (p.der_generation_kw - p.der_spill_kw) * dt,
         "DER consumed energy cost", maximums, "cash")
    near(ledger.external_cost, rewards.external_cost_per_period, "external per-period cost")
    near(ledger.ess_cost, rewards.ess_throughput_rate_per_kwh * (uc.charge_kw + p.ess_discharge_kw) * dt,
         "storage throughput cost", maximums, "cash")
    near(ledger.curtailment_cost, rewards.curtailment_rate_per_kwh * p.der_spill_kw * dt, "spill cost")
    near(ledger.dr_value, rewards.dr_value_rate_per_kwh * net * dt, "DR utility")
    near(ledger.terminal_energy_value,
         rewards.terminal_energy_value_per_kwh * (p.ess_energy_after_kwh - before.initial_ess_energy_kwh)
         if record.terminated else 0.0, "terminal stock value")
    near(ledger.terminal_pending_cost, rewards.uc_terminal_pending_rate_per_kwh
         * math.fsum(service.backlog_after_kwh for service in record.services.values())
         if record.terminated else 0.0, "terminal UC queue responsibility")
    near(ledger.cash_profit, total_bill - ledger.grid_cost - ledger.der_operating_cost - ledger.external_cost, "cash conservation", maximums, "cash")
    near(record.rewards["uc"], ledger.cash_profit - ledger.ess_cost - ledger.curtailment_cost
         + ledger.dr_value - ledger.uc_service_cost + ledger.terminal_energy_value - ledger.terminal_pending_cost,
         "UC expanded reward", maximums, "reward")
    after = engine.state
    assert after.period_id == before.period_id + 1
    for old, new, order in zip(before.consumers, after.consumers, record.orders):
        expected_month = old.monthly_uc_energy_kwh + record.market.allocations[old.consumer_id].uc_kw * dt
        near(record.monthly_closing_energy_kwh[old.consumer_id], expected_month, "actual UC monthly energy", maximums, "energy")
        near(new.monthly_uc_energy_kwh, 0.0 if record.month_rolled else expected_month, "monthly rollover")
    assert not record.terminated or config.system.episode_mode == "finite"
    coverage["charge_periods"] += int(uc.charge_kw > 1e-8)
    coverage["discharge_periods"] += int(uc.ess_discharge_offer_kw > 1e-8)
    coverage["idle_periods"] += int(uc.charge_kw <= 1e-8 and uc.ess_discharge_offer_kw <= 1e-8)
    coverage["negative_signed_DR"] += int(math.fsum(signed) < -1e-8)
    coverage["event_on"] += int(record.offer.event_active)
    coverage["event_off"] += int(not record.offer.event_active)
    coverage["zero_price_event"] += int(record.offer.event_active and record.offer.dr_incentive == 0)
    coverage["monthly_rollover"] += int(record.month_rolled)
    coverage["true_terminal"] += int(record.terminated)
    coverage["native_shortage"] += int(native_total > 1e-8)
    coverage["full_service"] += int(native_total < 1e-8)
    coverage["exact_bid_ties"] += int(len({a.der_bid_per_kwh for a in actions.values()}) < len(actions))
    coverage[f"tariff_band_{tariff_band(before.timestamp, config)}"] += 1
    coverage["empty_tariff_schedule"] += int(not market.uc_high_intervals_hours and not market.uc_peak_intervals_hours)
    coverage["multiple_tariff_intervals"] += int(len(market.uc_high_intervals_hours) > 1 or len(market.uc_peak_intervals_hours) > 1)
    coverage["fractional_tariff_hours"] += int(any(value != int(value) for interval in
        market.uc_high_intervals_hours + market.uc_peak_intervals_hours for value in interval))


_ORACLE_CHECKS = None


def compare_oracle(engine, before, record):
    global _ORACLE_CHECKS
    if _ORACLE_CHECKS is None:
        directory = ROOT / "environments/sn_energy_market/tests"
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location("v9_checks", directory / "v9_checks.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _ORACLE_CHECKS = module
    return _ORACLE_CHECKS.compare_record(engine, before, record)


def execute_period(engine, rng, index, coverage, maximums, *, oracle=False, snapshot=False, fixed=False):
    before, uc_obs = engine.state, engine.uc_observation()
    context = {"period": before.period_id, "state_before": before, "config": engine.config.to_dict()}
    try:
        uc = sample_uc(before, engine.profiles.der_kw[before.period_id], engine.config, rng, index, fixed)
        actions = sample_consumers(before, engine.profiles, engine.config, rng, index, fixed)
        context.update(uc_action=uc, consumer_actions=actions)
        observations = engine.apply_uc_action(uc)
        assert engine.state == before, "UC announcement advanced physical state"
        check_observations(engine, before, uc_obs, observations, actions, maximums)
        saved = engine.get_state() if snapshot else None
        record = engine.apply_consumer_actions(actions)
        context["record"] = record
        check_record(engine, before, uc, actions, record, coverage, maximums)
        if oracle:
            result = compare_oracle(engine, before, record)
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        maximums[f"oracle_{key}"] = max(maximums.get(f"oracle_{key}", 0.0), value)
            coverage["oracle_comparisons"] += 1
        if saved is not None:
            from snmarket import EnergyMarketEngine
            restored = EnergyMarketEngine(engine.config, engine.profiles)
            restored.set_state(saved)
            resumed = restored.apply_consumer_actions(actions)
            assert resumed == record and restored.state == engine.state
            coverage["snapshot_continuations"] += 1
        return record
    except Exception as exc:
        exc.validation_context = context
        raise


def reject_invalid(engine, rng, index, coverage):
    from snmarket import ConsumerAction, UCAction

    before = engine.state
    valid = sample_uc(before, engine.profiles.der_kw[0], engine.config, rng, index)
    consumers = sample_consumers(before, engine.profiles, engine.config, rng, index)
    selection = index % 12
    if selection < 8:
        values = list(dataclasses.astuple(valid))
        if selection == 0:
            values[0] = float("nan")
        elif selection == 1:
            values[1] = float("inf")
        elif selection == 2:
            values[0] = -1.0
        elif selection == 3:
            values[0] = engine.config.system.grid_import_max_kw + 1.0
        elif selection == 4:
            values[1] = float(engine.profiles.der_kw[0]) + 1.0
        elif selection == 5:
            values[3] = values[0] + 1.0
        elif selection == 6:
            values[4] = engine.config.system.ess_discharge_max_kw + 1.0
        else:
            values[5] = engine.config.market.dr_procurement_max_kw + 1.0
        invalid = UCAction(**dict(zip(('grid_import_offer_kw', 'der_planned_curtailment_kw', 'der_to_ess_kw', 'grid_to_ess_kw', 'ess_discharge_offer_kw', 'dr_procurement_cap_kw'), values, strict=True)))
        saved = engine.get_state()
        try:
            engine.apply_uc_action(invalid)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"Invalid UC action accepted: {invalid}")
        assert engine.get_state() == saved, "Invalid UC action mutated state/stage"
    else:
        engine.apply_uc_action(valid)
        saved = engine.get_state()
        identity = next(iter(consumers))
        values = list(dataclasses.astuple(consumers[identity]))
        values[selection - 8] = (float("nan"), float("inf"), 2.0, -1.0)[selection - 8]
        consumers[identity] = ConsumerAction(**dict(zip(('curtail_kw', 'signed_shift_kw', 'der_preference_fraction', 'der_bid_per_kwh'), values, strict=True)))
        try:
            engine.apply_consumer_actions(consumers)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"Invalid Consumer action accepted: {consumers[identity]}")
        assert engine.get_state() == saved, "Invalid Consumer action mutated state/stage"
    coverage[f"invalid_case_{selection}"] += 1


def execute_batch(task, size_function):
    coverage, totals, maximums = Counter(), Counter(), {}
    for index in range(task["start"], task["start"] + task["count"]):
        kind = task["kind"]
        count = size_function(index)
        if kind == "long":
            count = (2, 8, 32, 128)[index % 4]
        elif kind == "month":
            count = (2, 8, 32)[index % 3]
        periods = task["periods"] if kind in {"episode", "long", "month"} else 2
        engine, rng = scenario(task["seed"], index, count, periods, kind)
        all_profiles = engine.profiles
        if kind in {"long", "month"}:
            from snmarket import EnergyMarketEngine
            initial_state = engine.state
            engine = EnergyMarketEngine(engine.config, slice_profiles(all_profiles, 0, min(96, periods)))
            engine.reset(seed=task["seed"] + index, options={"consumer_states": initial_state.consumers,
                         "episode_id": initial_state.episode_id, "ess_energy_kwh": initial_state.ess_energy_kwh})
        try:
            if kind == "invalid":
                reject_invalid(engine, rng, index, coverage)
                totals["invalid_cases"] += 1
            else:
                run_periods = 1 if kind == "static" else periods
                for period in range(run_periods):
                    if period == len(engine.profiles.der_kw):
                        saved_state, saved_phase, saved_record = engine.state, engine.phase, engine.latest_record
                        try:
                            engine.apply_uc_action(sample_uc(engine.state, 0.0, engine.config, rng, index))
                        except Exception as exc:
                            if type(exc).__name__ != "ProfileExhaustedError":
                                raise
                        else:
                            raise AssertionError("Exhausted continuous buffer accepted a new UC announcement")
                        assert engine.state == saved_state and engine.phase == saved_phase
                        assert engine.latest_record == saved_record
                        engine.append_profiles(slice_profiles(all_profiles, period, min(period + 96, periods)))
                        assert engine.state == saved_state and engine.phase == saved_phase
                        assert engine.latest_record == saved_record
                        coverage["continuous_buffer_append"] += 1
                    execute_period(engine, rng, index * 101 + period, coverage, maximums,
                        oracle=kind == "static", snapshot=(index % 100 in {0, 20, 40, 80, 98}
                            and period == min(1, run_periods - 1)))
                    totals["physical_periods"] += 1
                    totals["static_physical_periods" if kind == "static" else "dynamic_physical_periods"] += 1
                    totals["consumer_records"] += count
                totals["static_cases" if kind == "static" else "episodes"] += 1
                coverage[f"size_{count}_episodes"] += 1
        except Exception as exc:
            context = getattr(exc, "validation_context", {})
            context.update(scenario_index=index, profile_seed=[task["seed"], kind, index],
                           profile_sha256=engine.profiles.fingerprint(), consumers=count, periods=periods,
                           config=engine.config.to_dict())
            exc.validation_context = context
            raise
    return {"totals": dict(totals), "coverage": dict(coverage), "maximums": maximums}


def slice_profiles(profiles, start, stop):
    from snmarket import Profiles
    return Profiles(profiles.loads_kw[start:stop], profiles.der_kw[start:stop],
                    profiles.grid_cost_per_kwh[start:stop], profiles.flex_kw[start:stop],
                    rigid_kw=profiles.rigid_kw[start:stop],
                    start_datetime=profiles.timestamp(start).isoformat(),
                    resolution_minutes=profiles.resolution_minutes, metadata=profiles.metadata)
