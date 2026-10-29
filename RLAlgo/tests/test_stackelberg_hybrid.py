"""Hybrid storage feasibility, conditional likelihood and finite evaluation."""

from dataclasses import astuple, replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.distribution.torch.torch_distribution import TorchMultiDistribution
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from snmarket import EnergyMarketEngine, EnvironmentConfig, Profiles
from snmarket.action import map_uc_controls, uc_control_bounds
from snmarket.uc import validate_uc_action

from raymarl.algorithms.stackelberg.rl_module import StackelbergRLModule


def engine_at(energy, generation=8.0, grid=10.0):
    config = EnvironmentConfig()
    config = replace(config, system=replace(
        config.system, num_consumers=1, horizon_steps=1,
        ess_energy_capacity_kwh=10.0, ess_initial_energy_kwh=energy,
        ess_charge_max_kw=5.0, ess_discharge_max_kw=5.0, grid_import_max_kw=grid,
    )).validate()
    profiles = Profiles(np.array([[4.0]]), np.array([generation]), np.array([0.35]),
                        np.zeros((1, 1)), start_datetime=config.system.start_datetime,
                        resolution_minutes=config.system.resolution_minutes)
    engine = EnergyMarketEngine(config, profiles)
    engine.reset()
    return engine


def uc_module():
    return RLModuleSpec(
        module_class=StackelbergRLModule,
        observation_space=gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
        action_space=gym.spaces.Box(-np.inf, np.inf, (6,), np.float64),
        model_config={"role": "uc", "hidden_dim": 8, "critic_dim": 7, "uc_head_scale": 0.0,
                      "uc_initial_mean": [0.8, 0.1, 0.4, 0.6, 0.5, 0.5]},
    ).build()


@pytest.mark.parametrize("energy,generation,grid", [
    (0.0, 8.0, 10.0), (10.0, 8.0, 10.0), (5.0, 8.0, 10.0),
    (0.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 10.0),
])
def test_sampled_modes_and_power_match_physical_environment(energy, generation, grid):
    torch.manual_seed(71)
    engine, module = engine_at(energy, generation, grid), uc_module()
    bounds = uc_control_bounds(engine.state, generation, engine.config)
    obs = torch.zeros(256, 7)
    with torch.no_grad():
        sample = module.draw(obs, torch.tensor(np.tile(bounds, (256, 1))))
    if energy == 0:
        assert not (sample["storage_mode"] == 2).any()
    if energy == 10:
        assert not (sample["storage_mode"] == 1).any()
    for index, mode in enumerate(sample["storage_mode"].tolist()):
        raw = sample["raw_latents"][index].numpy()
        action = map_uc_controls(np.r_[mode, raw], engine.state, generation, engine.config)
        validate_uc_action(action, engine.state, generation, engine.config)
        np.testing.assert_allclose(astuple(action), sample["actions"][index].numpy(), atol=1e-12, rtol=0)
        if mode == 0:
            assert astuple(action)[2:5] == (0.0, 0.0, 0.0)
    torch.testing.assert_close(module.evaluate_actions({"obs": obs, **sample})["action_logp"],
                               sample["action_logp"], atol=0, rtol=0)


def test_charge_sources_have_separate_power_controls_and_no_signed_gate():
    engine = engine_at(0.0)
    def mapped(der, grid):
        return map_uc_controls([1, 1, 0, der, grid, 0.9, 0], engine.state, 8, engine.config)
    assert astuple(mapped(0.4, 0))[2:5] == (2.0, 0.0, 0.0)
    assert astuple(mapped(0, 0.4))[2:5] == (0.0, 2.0, 0.0)
    assert astuple(mapped(0.4, 0.5))[2:5] == (2.0, 1.5, 0.0)
    with pytest.raises(ValueError, match="unavailable"):
        map_uc_controls([2, 1, 0, 0, 0, 1, 0], engine.state, 8, engine.config)
    with pytest.raises(ValueError, match="unavailable"):
        map_uc_controls([1, 0, 1, 1, 1, 0, 0], engine.state, 8, engine.config)


def test_ppo_reuses_stored_mode_and_prefix_without_resampling():
    module, obs = uc_module(), torch.zeros(64, 7)
    bounds = torch.tensor([[10., 8., 5., 5., 4., 0.]] * len(obs))
    with torch.no_grad():
        sample = module.draw(obs, bounds)
        module.actor["mode"].bias.add_(torch.tensor([2., -1., 0.5]))
    rng = torch.get_rng_state().clone()
    result = module.evaluate_actions({"obs": obs, **sample})
    assert torch.equal(rng, torch.get_rng_state())
    dist, _, _ = module._uc_distribution(module._features(obs), sample["raw_latents"],
                                         bounds.double(), sample["storage_mode"])
    assert isinstance(dist, TorchMultiDistribution)
    # Independent density calculation includes exactly the stored active coordinates.
    features, raw, mode = module._features(obs), sample["raw_latents"], sample["storage_mode"]
    base = module._uc_base(features, bounds)
    categorical, _ = module._uc_mode(features, raw[:, [0, 1, 5]], bounds)
    parameters = module._uc_storage_parameters(features, raw[:, [0, 1, 5]], mode).double()
    mean, logstd = parameters.chunk(2, -1)
    active = ~sample["zero_width_mask"][:, [2, 3, 4]]
    storage_logp = torch.distributions.Normal(mean, logstd.clamp(-5, 2).exp()).log_prob(raw[:, [2, 3, 4]])
    expected = base.logp(raw[:, [0, 1, 5]]) + categorical.logp(mode)
    expected = expected + torch.where(active, storage_logp, 0).sum(-1)
    torch.testing.assert_close(result["action_logp"], expected, atol=1e-12, rtol=0)
    assert not torch.allclose(result["action_logp"], sample["action_logp"])
    (-result["action_logp"].mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())


def test_inactive_storage_coordinates_do_not_update_storage_head():
    module, obs = uc_module(), torch.zeros(16, 7)
    with torch.no_grad():
        module.actor["mode"].bias.copy_(torch.tensor([100., -100., -100.]))
        sample = module.draw(obs, torch.tensor([[10., 8., 5., 5., 4., 1.]] * len(obs)))
    assert (sample["storage_mode"] == 0).all()
    original = module.evaluate_actions({"obs": obs, **sample})["action_logp"]
    changed = sample["raw_latents"].clone()
    changed[:, 2:5] = 123.0
    actual = module.evaluate_actions({"obs": obs, **sample, "raw_latents": changed})["action_logp"]
    torch.testing.assert_close(original, actual, atol=0, rtol=0)
    (-actual.mean()).backward()
    assert torch.count_nonzero(module.actor["storage"].weight.grad) == 0
    assert torch.count_nonzero(module.actor["storage"].bias.grad) == 0


def test_all_zero_domain_has_only_idle_and_finite_entropy_gradients():
    module, obs, bounds = uc_module(), torch.zeros(8, 7), torch.zeros(8, 6)
    sample = module.draw(obs, bounds)
    assert sample["mode_mask"].tolist() == [[True, False, False]] * 8
    assert torch.count_nonzero(sample["actions"]) == 0
    assert torch.count_nonzero(sample["action_logp"]) == 0
    entropy = module.policy_entropy({"obs": obs, "bounds": bounds})
    assert torch.count_nonzero(entropy) == 0
    entropy.mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
