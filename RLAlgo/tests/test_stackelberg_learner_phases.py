"""Policy priors, fixed value units and independent online update phases."""

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
from raymarl.algorithms.stackelberg.runner import generalized_advantages, random_state

from .test_stackelberg_policy import make_learner, sample_batch


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
    bounds = torch.tensor([[10., 8., 5., 5., 4., 3.]] * 8)
    distribution = module._uc_base(module._features(obs), bounds)
    torch.testing.assert_close(distribution.loc, torch.tensor(means, dtype=torch.float64)[[0, 1, 5]].expand(8, 3))
    torch.testing.assert_close(distribution._dist.scale, torch.tensor(stds, dtype=torch.float64)[[0, 1, 5]].expand(8, 3))
    draw = module.draw(obs, bounds)
    torch.testing.assert_close(draw["action_logp"], module.evaluate_actions({"obs": obs, **draw})["action_logp"])
    with torch.no_grad():
        module.actor["head"].bias[0] = -0.25
    assert module.draw(obs, bounds, deterministic=True)["raw_latents"][0, 0] == -0.25


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
    batch = {"critic_obs": torch.randn(16, 13)}
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


@pytest.mark.parametrize("terminal_at", [191, 383, None])
@pytest.mark.parametrize("segment_cut", [False, True])
@pytest.mark.parametrize("freeze_uc", [False, True])
def test_gae_value_source_boundaries_and_no_rng(terminal_at, segment_cut, freeze_uc):
    from experiments.sn_energy_market.training_audit import (
        reference_advantages,
        reference_refreshed_advantages,
    )

    torch.set_num_threads(1)
    torch.manual_seed(72)
    learner = make_learner()
    learner.config.gamma, learner.config.lambda_gae = 0.99, 0.97
    learner.config.uc_frozen_gae = freeze_uc
    batch = MultiAgentBatch({key: sample_batch(learner.module[key], 384) for key in ("uc", "consumer")}, 384)
    for key, data in batch.policy_batches.items():
        data["next_value_obs"] = np.roll(data["critic_obs"], -1, axis=0).copy()
        data["rewards"] = np.linspace(-0.2, 0.3, 384, dtype=np.float64)
        data["terminateds"] = np.zeros(384, dtype=bool)
        data["segment_end"] = np.zeros(384, dtype=bool)
        data["next_vf_preds"] = np.roll(data["vf_preds"], -1).copy()
        if segment_cut:
            data["segment_end"][191] = True
        if terminal_at is not None:
            data["terminateds"][terminal_at] = True
            data["next_vf_preds"][terminal_at] = 0.0
    original = copy.deepcopy(dict(batch.policy_batches))
    old = learner.recompute_advantages(batch)
    learner.update(batch=batch, update_phase="critic", num_epochs=4, minibatch_size=32)
    fitted, rng = copy.deepcopy(learner.get_state()), random_state()
    fresh = learner.recompute_advantages(batch)
    expected = reference_refreshed_advantages(learner, batch)
    if freeze_uc:
        expected["uc"] = reference_advantages(batch["uc"], 0.99, 0.97)
    assert_nested_equal(fitted, learner.get_state())
    assert_nested_equal(rng, random_state())
    assert_nested_equal(original, dict(batch.policy_batches))
    for key in fresh:
        np.testing.assert_allclose(fresh[key], expected[key], atol=1e-12, rtol=1e-12)
        assert np.allclose(old[key], fresh[key]) == (freeze_uc and key == "uc")
    # A later reward may cross an ordinary sampling unit, but cannot cross
    # a true terminal or a boundary between independent streams.
    for data in batch.policy_batches.values():
        data["rewards"][192] += 7
    changed = learner.recompute_advantages(batch)
    for key in fresh:
        difference = 0 if terminal_at == 191 or segment_cut else 7 * 0.99 * 0.97
        assert changed[key][191] - fresh[key][191] == pytest.approx(difference)
    # The exact next input saved at collection, rather than current RNG, sets the tail.
    torch.randn(100)
    np.random.random(10)
    again = learner.recompute_advantages(batch)
    assert_nested_equal(changed, again)
    for data in batch.policy_batches.values():
        data["next_value_obs"][-1] += 3
    tail_changed = learner.recompute_advantages(batch)
    for key in fresh:
        assert (tail_changed[key][-1] == again[key][-1]) == (
            terminal_at == 383 or (freeze_uc and key == "uc")
        )


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
    rng_before = random_state()
    diagnostics = learner.batch_diagnostics(batch, actor_ids=(active_actor,))
    assert_nested_equal(rng_before, random_state())
    assert_nested_equal(before, learner.get_state())
    for key in module_ids:
        expected_mse = np.mean((batch[key]["vf_preds"] - batch[key]["value_targets"]) ** 2)
        assert diagnostics[key]["critic_loss"] == pytest.approx(0.5 * expected_mse)
        assert diagnostics[key]["critic_rmse"] == pytest.approx(np.sqrt(expected_mse))
        assert (diagnostics[key]["policy_loss"] is None) == (key != active_actor)
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
    measured = learner.batch_diagnostics(batch, actor_ids=(active_actor,))
    for key in module_ids:
        assert measured[key]["critic_loss"] == pytest.approx(0.5 * measured[key]["critic_rmse"] ** 2)
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
