"""Configured policy priors and fixed-unit Value fitting without Actor updates."""

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch

from raymarl.algorithms.stackelberg.learner import StackelbergTorchLearner, value_fit_statistics
from raymarl.algorithms.stackelberg.rl_module import StackelbergRLModule
from raymarl.algorithms.stackelberg.runner import generalized_advantages

from .test_stackelberg_policy import sample_batch


def specification(role, **settings):
    return RLModuleSpec(
        module_class=StackelbergRLModule,
        observation_space=gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
        action_space=gym.spaces.Box(-np.inf, np.inf, (6 if role == "uc" else 4,), np.float64),
        model_config={"role": role, "hidden_dim": 12, "critic_dim": 13, **settings},
    )


def assert_nested_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        np.testing.assert_array_equal(left, right)


def test_configured_uc_distribution_keeps_raw_action_probability():
    means = [0.9, 0.02, 0.5, 0.5, 0.0, 0.5]
    stds = [0.12, 0.08, 0.15, 0.15, 0.1, 0.15]
    module = specification("uc", uc_initial_mean=means, uc_initial_std=stds, uc_head_scale=0.0).build()
    obs = torch.randn(8, 7)
    distribution = module._normal(module._features(obs))
    torch.testing.assert_close(distribution.mean, torch.tensor(means).expand(8, 6))
    torch.testing.assert_close(distribution.stddev, torch.tensor(stds).expand(8, 6))
    draw = module.draw(obs)
    torch.testing.assert_close(draw["action_logp"], distribution.log_prob(draw["raw_latents"]).sum(-1))
    torch.testing.assert_close(module.draw(obs, deterministic=True)["actions"], distribution.mean)
    # The configured prior is initialization, not a bound or a pinned action.
    with torch.no_grad():
        module.actor["head"].bias[0] = -0.25
    assert module.draw(obs, deterministic=True)["actions"][0, 0] == -0.25


@pytest.mark.parametrize("std", [0.0, -0.1, [0.1] * 5, float("nan")])
def test_invalid_policy_std_is_rejected(std):
    with pytest.raises(ValueError, match="initialization"):
        specification("uc", uc_initial_std=std).build()


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf")])
def test_invalid_value_unit_is_rejected(scale):
    with pytest.raises(ValueError, match="value_scale"):
        specification("uc", value_scale=scale).build()


def test_value_units_preserve_gae_and_normalized_loss_gradients():
    torch.manual_seed(19)
    unit = specification("uc").build()
    scaled = specification("uc", value_scale={"uc": 50.0, "consumer": 3.0}).build()
    scaled.set_state(unit.get_state())
    batch = {"obs": torch.randn(16, 7)}
    targets = torch.linspace(-3.0, 0.0, 16)
    prediction = unit.compute_values(batch)
    scaled_prediction = scaled.compute_values(batch)
    torch.testing.assert_close(scaled_prediction, 50 * prediction)
    loss, rmse, explained = value_fit_statistics(prediction, targets, 1.0)
    scaled_loss, scaled_rmse, scaled_explained = value_fit_statistics(scaled_prediction, 50 * targets, 50.0)
    torch.testing.assert_close(loss, scaled_loss)
    torch.testing.assert_close(scaled_rmse, 50 * rmse)
    torch.testing.assert_close(explained, scaled_explained)
    loss.backward()
    scaled_loss.backward()
    for first, second in zip(unit.value.parameters(), scaled.value.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad)
    reward, values = targets.numpy(), prediction.detach().numpy()
    args = (np.zeros(16, dtype=bool), 0.98, 0.9)
    advantage, returns = generalized_advantages(reward, values, values / 2, *args)
    large_advantage, large_returns = generalized_advantages(50 * reward, 50 * values, 25 * values, *args)
    np.testing.assert_allclose(large_advantage, 50 * advantage, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(large_returns, 50 * returns, rtol=1e-6, atol=1e-5)
    assert torch.isfinite(value_fit_statistics(prediction.detach(), torch.ones(16), 1)[2])


def test_value_only_fit_improves_targets_and_preserves_actor_optimizer_and_restore():
    torch.set_num_threads(1)
    torch.manual_seed(25)
    config = AlgorithmConfig().learners(num_learners=0, num_gpus_per_learner=0)
    config.actor_lr, config.critic_lr = 0.001, 0.01
    config.clip_param, config.vf_loss_coeff, config.entropy_coeff, config.grad_clip = 0.2, 0.5, 0.0, 0.5
    learner = StackelbergTorchLearner(
        config=config,
        module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "uc": specification("uc", value_scale=20.0),
                "consumer": specification("consumer", value_scale=3.0),
            }
        ),
    )
    learner.build()
    learner.update(
        batch=MultiAgentBatch({key: sample_batch(learner.module[key]) for key in ("uc", "consumer")}, 16),
        num_epochs=1,
        minibatch_size=16,
    )
    obs = torch.randn(32, 7, requires_grad=True)
    targets = -40.0 + 3.0 * obs[:, 0].tanh()
    batches = {"uc": {"obs": obs, "critic_obs": obs, "value_targets": targets}}
    actor = copy.deepcopy(learner.module["uc"].actor.state_dict())
    actor_optimizer = copy.deepcopy(learner.get_optimizer("uc", "actor").state_dict())
    consumer = copy.deepcopy(learner.module["consumer"].get_state())
    consumer_optimizers = {
        name: copy.deepcopy(learner.get_optimizer("consumer", name).state_dict())
        for name in ("actor", "value")
    }
    snapshot, rng = copy.deepcopy(learner.get_state()), torch.get_rng_state()
    report = learner.fit_values(batches, epochs=30, minibatch_size=16)
    assert obs.grad is None
    assert report["optimizer_steps"] == 60
    metrics = report["modules"]["uc"]
    assert metrics["rmse_after"] < 0.15 * metrics["rmse_before"]
    assert_nested_equal(actor, learner.module["uc"].actor.state_dict())
    assert_nested_equal(actor_optimizer, learner.get_optimizer("uc", "actor").state_dict())
    assert_nested_equal(consumer, learner.module["consumer"].get_state())
    for name in ("actor", "value"):
        assert_nested_equal(consumer_optimizers[name], learner.get_optimizer("consumer", name).state_dict())
    expected = copy.deepcopy(learner.get_state())
    learner.set_state(snapshot)
    torch.set_rng_state(rng)
    repeated = learner.fit_values(batches, epochs=30, minibatch_size=16)
    assert repeated == report
    assert_nested_equal(expected, learner.get_state())


@pytest.mark.parametrize("active_actor", ["uc", "consumer"])
def test_online_phases_preserve_frozen_adam_and_fixed_targets(active_actor):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    config = AlgorithmConfig().learners(num_learners=0, num_gpus_per_learner=0)
    config.actor_lr, config.critic_lr = 0.001, 0.01
    config.clip_param, config.vf_loss_coeff, config.entropy_coeff, config.grad_clip = 0.2, 0.5, 0.03, 0.5
    learner = StackelbergTorchLearner(
        config=config,
        module_spec=MultiRLModuleSpec(rl_module_specs={
            "uc": specification("uc"), "consumer": specification("consumer"),
        }),
    )
    learner.build()
    # Populate moments first: checking fresh Adam would miss zero-gradient drift.
    module_ids = ("uc", "consumer")
    batch = MultiAgentBatch({key: sample_batch(learner.module[key]) for key in module_ids}, 16)
    learner.update(batch=batch, num_epochs=1, minibatch_size=16)
    batch = MultiAgentBatch({key: sample_batch(learner.module[key]) for key in module_ids}, 16)
    fixed = copy.deepcopy(dict(batch.policy_batches))
    before = copy.deepcopy(learner.get_state())
    learner.update(batch=batch, update_phase="critic", num_epochs=2, minibatch_size=16)
    fitted = copy.deepcopy(learner.get_state())
    for key in module_ids:
        for name, value in before["rl_module"][key].items():
            if name.startswith("actor."):
                np.testing.assert_array_equal(value, fitted["rl_module"][key][name])
        assert any(
            not np.array_equal(value, fitted["rl_module"][key][name])
            for name, value in before["rl_module"][key].items() if name.startswith("value.")
        )
    for name, state in before["optimizer"].items():
        if name.endswith("actor"):
            assert_nested_equal(state, fitted["optimizer"][name])
    learner.update(
        batch=MultiAgentBatch({active_actor: batch[active_actor]}, 16),
        update_phase="actor", num_epochs=1, minibatch_size=16,
    )
    acted = learner.get_state()
    for key in module_ids:
        for name, value in fitted["rl_module"][key].items():
            if key != active_actor or name.startswith("value."):
                np.testing.assert_array_equal(value, acted["rl_module"][key][name])
    for name, state in fitted["optimizer"].items():
        if name.endswith("value") or state["module_id"] != active_actor:
            assert_nested_equal(state, acted["optimizer"][name])
    assert any(
        not np.array_equal(value, acted["rl_module"][active_actor][name])
        for name, value in fitted["rl_module"][active_actor].items() if name.startswith("actor.")
    )
    for key, rows in fixed.items():
        for name in ("advantages", "value_targets", "action_logp"):
            np.testing.assert_array_equal(rows[name], batch[key][name])
    assert learner._update_phase == "joint"
    with pytest.raises(ValueError, match="update_phase"):
        learner.update(batch=batch, update_phase="unknown")
    # Failure inside the native update must also clear phase ownership.
    with pytest.raises(ValueError):
        learner.update(update_phase="actor")
    assert learner._update_phase == "joint"
