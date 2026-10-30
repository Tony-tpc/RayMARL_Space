"""Causal aggregate supply/demand summaries for the UC value function only."""

import numpy as np


def physical_summaries(profiles, system, history):
    """24/72 h persistence forecasts using only current and preceding readings.

    Missing weekly readings fall back to yesterday's same slot; missing daily
    readings fall back to the current reading. No synthetic prefix is stored.
    Once seven days are available this is exactly the offline summary formula.
    """
    cadence = profiles.resolution_minutes
    if 1440 % cadence:
        raise ValueError("Summary cadence must divide a day")
    day = 1440 // cadence
    history = np.asarray(history, dtype=np.float64)
    if (history.ndim != 2 or history.shape[1] != 2 or len(history) > 7 * day
            or not np.isfinite(history).all() or (history < 0).any()):
        raise ValueError("History must contain at most seven days of aggregate load/PV")
    current = np.column_stack((profiles.loads_kw.sum(axis=1), profiles.der_kw))
    raw = np.concatenate((history, current))
    t = np.arange(len(history), len(raw))[:, None]
    k = np.arange(3 * day)[None, :]
    daily_index, weekly_index = t - day + k % day, t - 7 * day + k
    daily = np.where((daily_index >= 0)[..., None], raw[daily_index.clip(0)], raw[t])
    weekly = np.where((weekly_index >= 0)[..., None], raw[weekly_index.clip(0)], daily)
    forecast = (daily + weekly) * 0.5
    forecast[:, 0] = current
    columns = []
    for length in (day, 3 * day):
        load, pv = forecast[:, :length, 0], forecast[:, :length, 1]
        gap = (1 - system.dr_limit_ratio) * load - pv - system.grid_import_max_kw
        columns.extend((load.sum(axis=1) * system.dt_hours,
                        pv.sum(axis=1) * system.dt_hours,
                        gap.clip(0).sum(axis=1) * system.dt_hours, gap.max(axis=1)))
    values = np.column_stack(columns)
    # Immutable backing bytes let candidate snapshots share this derived table.
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(-1, 8)
