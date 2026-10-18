"""Native-first service and earliest-deadline/FIFO repayment of actual energy."""

import math
from dataclasses import replace

from .contracts import LoadService, ShiftTask


def realize_service(order, allocation, consumer, period, config, index=0):
    served = allocation.served_kw
    if not math.isfinite(served) or served < 0.0 or served > order.request_kw + 1e-8:
        raise ValueError("Actual service exceeds the submitted request")
    dt = config.system.dt_hours
    lost = max(0.0, order.native_kw - served)
    repayment = max(0.0, served - order.native_kw)
    requested_repayment = max(0.0, -order.action.signed_shift_kw)
    amount = repayment * dt
    tasks = []
    for _, task in sorted(enumerate(consumer.tasks),
                          key=lambda pair: (pair[1].deadline_period, pair[1].source_period, pair[0])):
        used = min(task.remaining_kwh, amount)
        amount -= used
        remainder = task.remaining_kwh - used
        # Fully served tasks can retain an ulp after subtracting native service.
        # Scale the zero threshold to floating arithmetic, not to a fixed energy
        # quantum: a genuine small task with no repayment is always retained.
        rounding = 32 * max(math.ulp(task.remaining_kwh), math.ulp(served) * dt,
                            math.ulp(order.native_kw) * dt, math.ulp(repayment * dt))
        if used > 0.0 and 0.0 < remainder <= rounding:
            remainder = 0.0
        if remainder > 0.0:
            tasks.append(replace(task, remaining_kwh=remainder))
    shift_out = max(0.0, order.action.signed_shift_kw)
    if shift_out > 0.0:
        task_id = f"{consumer.consumer_id}:shift:{period}"
        existing_ids = {task.task_id for task in consumer.tasks}
        suffix = 0
        while task_id in existing_ids:
            suffix += 1
            task_id = f"{consumer.consumer_id}:shift:{period}:{suffix}"
        tasks.append(ShiftTask(period, period + config.market.shift_deadline_steps, shift_out * dt, task_id))
    tasks.sort(key=lambda task: (task.deadline_period, task.source_period))
    backlog = math.fsum(task.remaining_kwh for task in tasks)
    due_left = math.fsum(task.remaining_kwh for task in tasks if task.deadline_period <= period)
    request_residual = order.request_kw - served - lost - (requested_repayment - repayment)
    original_residual = (order.baseline_kw - order.action.curtail_kw - shift_out + repayment
                         - served - lost)
    queue_residual = backlog - consumer.backlog_kwh - shift_out * dt + repayment * dt
    return LoadService(
        consumer.consumer_id, min(order.native_kw, served), lost, repayment,
        max(0.0, requested_repayment - repayment), backlog, due_left,
        max(0.0, order.due_kwh - order.flex_kw * dt), tuple(tasks),
        request_residual, original_residual, queue_residual,
    )

