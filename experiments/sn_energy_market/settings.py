"""Shared model priors and terminal inventory valuation for the energy experiment."""

from dataclasses import replace


def model_settings(hidden_dim):
    """Experiment priors and value units; physical actions remain freely sampled."""
    return {
        "hidden_dim": hidden_dim,
        "uc_initial_mean": [0.95, 0.02, 0.2, 0.2, 0.5, 0.5],
        "uc_initial_std": [0.08, 0.05, 0.2, 0.2, 0.2, 0.2],
        "uc_head_scale": 0.01,
        "value_scale": {"uc": 20.0, "consumer": 10.0},
    }


def inventory_valued_config(config):
    """Value stored energy at the fixed off-peak reference after output costs."""
    if config.rewards.terminal_energy_value_per_kwh > 0:
        return config
    value = config.system.ess_discharge_efficiency * max(
        0.0, 0.35 - config.rewards.ess_throughput_rate_per_kwh
    )
    return replace(config, rewards=replace(config.rewards, terminal_energy_value_per_kwh=value)).validate()
