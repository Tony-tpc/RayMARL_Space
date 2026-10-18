from dataclasses import replace

import numpy as np
import pytest

from snmarket import EnergyMarketEngine, EnvironmentConfig, Profiles


@pytest.fixture
def make_engine():
    def factory(loads, *, der=0.0, flex=0.0, rigid=None, grid=100.0,
                start="2026-01-01T00:00:00+08:00", deadline=4, active=True,
                ess=5.0, grid_cost=0.50, dt_minutes=15, mode="finite",
                rewards=None, market=None, system=None):
        loads = np.asarray(loads, dtype=float)
        if loads.ndim == 1:
            loads = loads[None, :]
        periods, count = loads.shape
        base = EnvironmentConfig()
        system_values = dict(num_consumers=count, horizon_steps=periods, grid_import_max_kw=grid,
                             ess_energy_capacity_kwh=10.0, ess_initial_energy_kwh=ess,
                             start_datetime=start, resolution_minutes=dt_minutes, episode_mode=mode)
        system_values.update(system or {})
        sys = replace(base.system, **system_values)
        market_values = dict(shift_deadline_steps=deadline,
                             dr_event_intervals=((0, sys.steps_per_day),) if active else ())
        market_values.update(market or {})
        config = replace(base, system=sys, market=replace(base.market, **market_values),
                         rewards=replace(base.rewards, **(rewards or {}))).validate()
        profiles = Profiles(loads, np.broadcast_to(der, (periods,)),
                            np.broadcast_to(grid_cost, (periods,)), np.broadcast_to(flex, loads.shape),
                            rigid_kw=None if rigid is None else np.broadcast_to(rigid, loads.shape),
                            start_datetime=start, resolution_minutes=dt_minutes)
        return EnergyMarketEngine(config, profiles)
    return factory
