import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from snmarket import EnvironmentConfig, Profiles


@pytest.mark.parametrize("group,field,value", [
    ("system", "resolution_minutes", 0), ("system", "resolution_minutes", 7),
    ("system", "num_consumers", True), ("system", "horizon_steps", 0),
    ("system", "episode_mode", "auto_repeat"), ("system", "grid_import_max_kw", -1),
    ("system", "ess_initial_energy_kwh", 300), ("system", "ess_charge_efficiency", 0),
    ("system", "ess_discharge_efficiency", 1.1), ("system", "start_datetime", "2026-01-01T00:00:00"),
    ("system", "start_datetime", "2026-01-01T00:01:00+08:00"),
    ("market", "shift_deadline_steps", 0), ("market", "shift_deadline_steps", True),
    ("market", "dr_procurement_max_kw", -1), ("market", "mechanism_version", "v5"),
    ("market", "tier_thresholds_kwh", (100, 100)), ("market", "uc_flat_rates", (0.4,)),
    ("market", "dr_event_intervals", ((0, 5), (4, 6))),
    ("market", "der_bid_cap", 0.2), ("market", "shift_backlog_limit_kwh", (1, 2)),
    ("rewards", "adjustment_epsilon_kw", 0), ("rewards", "uc_supply_shortfall_rate_per_kwh", -1),
    ("rewards", "consumer_native_loss_rate_per_kwh", (1, 2)),
    ("rewards", "external_cost_per_period", float("nan")),
    ("rewards", "terminal_energy_value_per_kwh", float("inf")),
])
def test_invalid_config_rejected(group, field, value):
    base = EnvironmentConfig()
    changed = replace(getattr(base, group), **{field: value})
    with pytest.raises(ValueError):
        replace(base, **{group: changed}).validate()


@pytest.mark.parametrize("group,legacy", [("system", "ess_decay_per_period"),
                                         ("market", "backup_capacity_limit_kw"),
                                         ("market", "uc_bid_cap"),
                                         ("rewards", "backup_activation_rate_per_kwh")])
def test_legacy_config_not_silently_reinterpreted(group, legacy):
    values = EnvironmentConfig().to_dict()
    values[group][legacy] = 0
    with pytest.raises(ValueError):
        EnvironmentConfig.from_dict(values)


def test_config_serialization_and_checked_in_configuration():
    config = EnvironmentConfig().validate()
    assert EnvironmentConfig.from_dict(config.to_dict()) == config
    directory = Path(__file__).parents[1] / "configs"
    loaded = EnvironmentConfig.from_directory(directory)
    assert loaded.market.mechanism_version.startswith("v9")
    assert loaded.rewards.actual_compensation_per_kwh == 1
    assert loaded.rewards.consumer_native_loss_rate_per_kwh == 2
    assert loaded.rewards.uc_supply_shortfall_rate_per_kwh == 2


def profiles(**changes):
    kwargs = dict(loads_kw=[[4], [5]], der_kw=[1, 2], grid_cost_per_kwh=[0.4, 0.7],
                  flex_kw=[[2], [2]], rigid_kw=[[2], [3]],
                  start_datetime="2026-01-01T00:00:00+08:00", resolution_minutes=15)
    kwargs.update(changes)
    return Profiles(**kwargs)


@pytest.mark.parametrize("changes", [dict(loads_kw=[]), dict(der_kw=[1]), dict(flex_kw=[[1, 2], [1, 2]]),
                                     dict(rigid_kw=[[5], [5]]), dict(rigid_kw=[[-1], [0]]),
                                     dict(grid_cost_per_kwh=[float("nan"), 1]),
                                     dict(flex_kw=[[float("inf")], [2]]),
                                     dict(start_datetime="2026-01-01T00:01:00+08:00"),
                                     dict(resolution_minutes=0)])
def test_bad_profiles_rejected(changes):
    with pytest.raises(ValueError):
        profiles(**changes)


def test_profile_arrays_are_copied_finite_and_read_only():
    source = np.array([[4.0], [5.0]])
    data = profiles(loads_kw=source)
    before = data.fingerprint()
    source[:] = 99
    assert data.loads_kw[:, 0].tolist() == [4, 5]
    for name in ("loads_kw", "der_kw", "grid_cost_per_kwh", "flex_kw", "rigid_kw"):
        with pytest.raises(ValueError):
            getattr(data, name).flat[0] = 100
    assert data.fingerprint() == before


def test_default_rigid_data_is_explicit_derivation():
    data = profiles(rigid_kw=None, flex_kw=[[1], [6]])
    assert data.rigid_kw[:, 0].tolist() == [3, 0]


def test_kernel_import_does_not_import_ray_or_torch():
    command = [sys.executable, "-B", "-c", "import sys; import snmarket; "
               "assert 'ray' not in sys.modules; assert 'torch' not in sys.modules; print('kernel-only')"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "kernel-only" in result.stdout
