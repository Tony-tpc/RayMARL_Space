"""Frozen UC retail tariff; monthly tiers count executed UC energy only."""

from bisect import bisect_right

from .config import nonnegative


def uc_price(timestamp, monthly_uc_energy_kwh, config):
    nonnegative(monthly_uc_energy_kwh, "monthly UC energy")
    tier = bisect_right(config.market.tier_thresholds_kwh, monthly_uc_energy_kwh)
    hour = timestamp.hour + timestamp.minute / 60
    high = any(start <= hour < end for start, end in config.market.uc_high_intervals_hours)
    peak = any(start <= hour < end for start, end in config.market.uc_peak_intervals_hours)
    rates = (config.market.uc_peak_rates if peak else
             config.market.uc_high_rates if high else config.market.uc_flat_rates)
    return float(rates[tier])


def event_context(timestamp, config):
    local_period = (timestamp.hour * 60 + timestamp.minute) // config.system.resolution_minutes
    for index, (start, end) in enumerate(config.market.dr_event_intervals):
        if start <= local_period < end:
            return True, f"{timestamp.date().isoformat()}:{index}"
    return False, ""
