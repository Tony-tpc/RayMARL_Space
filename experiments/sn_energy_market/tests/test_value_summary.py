from dataclasses import replace

import numpy as np
import pytest
import torch
from snmarket import EnvironmentConfig, Profiles

from experiments.sn_energy_market.value_summary import physical_summaries
from raymarl.algorithms.stackelberg.rl_module import ValueNetwork, signed_log1p


def inputs(raw):
    base = EnvironmentConfig()
    system = replace(base.system, resolution_minutes=30)
    profiles = Profiles(raw[:, :1], raw[:, 1], np.ones(len(raw)), raw[:, :1] * .2,
                        start_datetime="2020-07-01T00:00:00+10:00", resolution_minutes=30)
    return profiles, system


def reference(raw, t, system):
    forecast = []
    for k in range(144):
        daily = raw[t - 48 + k % 48] if t - 48 + k % 48 >= 0 else raw[t]
        weekly = raw[t - 336 + k] if t - 336 + k >= 0 else daily
        forecast.append((daily + weekly) / 2)
    forecast = np.array(forecast)
    forecast[0] = raw[t]
    result = []
    for length in (48, 144):
        load, pv = forecast[:length].T
        gap = .8 * load - pv - system.grid_import_max_kw
        result.extend([load.sum() * .5, pv.sum() * .5, np.maximum(gap, 0).sum() * .5, gap.max()])
    return result


def test_summary_matches_scalar_causal_formula_and_cold_start():
    raw = np.random.default_rng(11).uniform(0, 120, (720, 2))
    profiles, system = inputs(raw)
    actual = physical_summaries(profiles, system, np.empty((0, 2)))
    for t in (0, 1, 47, 48, 143, 335, 336, 719):
        np.testing.assert_allclose(actual[t], reference(raw, t, system), atol=1e-10)
        changed = raw.copy()
        changed[t+1:] += 10000
        np.testing.assert_array_equal(actual[:t+1], physical_summaries(inputs(changed)[0], system, np.empty((0, 2)))[:t+1])
    assert not actual.flags.writeable


def test_previous_month_history_matches_continuous_series():
    raw = np.random.default_rng(23).uniform(0, 120, (1000, 2))
    profiles, system = inputs(raw)
    full = physical_summaries(profiles, system, np.empty((0, 2)))
    later = physical_summaries(inputs(raw[700:])[0], system, raw[364:700])
    np.testing.assert_array_equal(full[700:], later)


def test_value_scaling_is_only_selected_columns_and_has_correct_gradients():
    value = ValueNetwork(76, 192, {0: 1488, 1: 1488})
    x = torch.ones(3, 76, requires_grad=True)
    transformed = value.transform(x)
    torch.testing.assert_close(transformed[:, :2], x[:, :2] / 1488, atol=0, rtol=0)
    torch.testing.assert_close(transformed[:, 2:], signed_log1p(x[:, 2:]), atol=0, rtol=0)
    transformed.sum().backward()
    torch.testing.assert_close(x.grad[:, :2], torch.full((3, 2), 1 / 1488))
    torch.testing.assert_close(x.grad[:, 2:], torch.full((3, 74), .5))
    consumer = ValueNetwork(76, 128)
    torch.testing.assert_close(consumer.transform(x), signed_log1p(x), atol=0, rtol=0)


@pytest.mark.parametrize("scales", [{-1: 1488}, {76: 1488}, {0: 0}, {0: float("nan")}])
def test_invalid_scaling_rejected(scales):
    with pytest.raises(ValueError):
        ValueNetwork(76, 192, scales)
