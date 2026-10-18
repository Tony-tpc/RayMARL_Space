"""Frozen entry tariff; only executed UC energy advances the monthly counter."""

from .contracts import ConsumerState


def next_consumer_state(consumer, allocation, service, month_rolled, dt_hours):
    closing = consumer.monthly_uc_energy_kwh + allocation.uc_kw * dt_hours
    return ConsumerState(consumer.consumer_id, 0.0 if month_rolled else closing, service.tasks_after), closing

