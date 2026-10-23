"""Synthetic SOC, passive gradient diagnostics and unchanged native PPO updates."""

import copy
import json

import numpy as np
import pytest
import torch
from ray.rllib.policy.sample_batch import MultiAgentBatch

from raymarl.algorithms.stackelberg.learner import clipped_policy_loss
from raymarl.algorithms.stackelberg.mode_diagnostics import flat_surrogate, mode_statistics
from raymarl.algorithms.stackelberg.rl_module import signed_log1p
from raymarl.algorithms.stackelberg.runner import random_state, restore_random_state

from .test_stackelberg_learner_phases import assert_nested_equal, specification
from .test_stackelberg_policy import make_learner, sample_batch


def test_soc_skip_adds_only_three_weights_preserves_critic_rng_and_raw_observation():
    torch.manual_seed(9)
    old = specification("uc").build()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(9)
    module = specification("uc", soc_input={"index": 2, "min": 2., "max": 12.}).build()
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in module.parameters()) - sum(p.numel() for p in old.parameters()) == 3
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, module.state_dict()[key], rtol=0, atol=0)
    assert torch.count_nonzero(module.actor["mode_soc"]["weight"]) == 0
    obs = torch.ones(3, 7)
    obs[:, 2] = torch.tensor([2., 7., 12.])
    original = obs.clone()
    torch.testing.assert_close(module.centered_soc(obs), torch.tensor([-1., 0., 1.]))
    expected = signed_log1p(obs)
    expected[:, 2] = torch.tensor([-1., 0., 1.])
    torch.testing.assert_close(module._features(obs), module.actor["encoder"](expected), atol=0, rtol=0)
    torch.testing.assert_close(module.compute_values({"obs": obs}), old.compute_values({"obs": obs}), atol=0, rtol=0)
    torch.testing.assert_close(obs, original, atol=0, rtol=0)


def test_soc_mode_connection_and_likelihood_replay_with_mask():
    module = specification("uc", soc_input={"index": 2, "min": 0., "max": 10.}, uc_head_scale=0.).build()
    obs = torch.zeros(3, 7)
    obs[:, 2] = torch.tensor([0., 5., 10.])
    bounds = torch.tensor([[10., 8., 5., 5., 4., 3.]] * 3, dtype=torch.float64)
    with torch.no_grad():
        module.actor["mode_soc"]["weight"][:, 0] = torch.tensor([0., -2., 2.])
        draw = module.draw(obs, bounds, deterministic=True)
        assert draw["storage_mode"].tolist() == [1, 0, 2]
        # Actual physical feasibility masks override even a strong SOC preference.
        bounds[:, 2] = 0
        bounds[:, 4] = 0
        draw = module.draw(obs, bounds)
        assert draw["storage_mode"].tolist() == [0, 0, 0]
        output = module.evaluate_actions({"obs": obs, **draw})
    torch.testing.assert_close(draw["action_logp"], output["action_logp"], atol=0, rtol=0)
    torch.testing.assert_close(draw["logp_parts"].sum(-1), draw["action_logp"], atol=0, rtol=0)
    torch.testing.assert_close(draw["logp_parts"], output["logp_parts"], atol=0, rtol=0)


def test_zero_capacity_has_finite_neutral_soc():
    module = specification("uc", soc_input={"index": 2, "min": 0., "max": 0.}).build()
    torch.testing.assert_close(module.centered_soc(torch.zeros(5, 7)), torch.zeros(5))


def test_soc_probe_measures_learned_interior_switch_without_mutating_rng_or_state():
    from experiments.sn_energy_market.audit_soc_mode import probe

    metadata = {"index": 2, "min": 0., "max": 10.}
    module = specification("uc", soc_input=metadata, uc_head_scale=0.).build()
    obs = torch.ones(4, 7)
    bounds = torch.tensor([[10., 8., 5., 5., 4., 3.]] * 4, dtype=torch.float64)
    base = torch.full((4, 3), .5)
    with torch.no_grad():
        module.actor["mode_soc"]["weight"][:, 0] = torch.tensor([0., -2., 2.])
    before, rng = copy.deepcopy(module.get_state()), random_state()
    observation, boundaries = obs.clone(), bounds.clone()
    _, result = probe(module, obs, bounds, base, metadata)
    assert result["states"] == 4 and result["switch_fraction"] == 1
    assert result["low_high_transitions"][1][2] == 4
    assert_nested_equal(before, module.get_state())
    assert_nested_equal(rng, random_state())
    torch.testing.assert_close(obs, observation, atol=0, rtol=0)
    torch.testing.assert_close(bounds, boundaries, atol=0, rtol=0)
    assert all(p.grad is None for p in module.parameters())


def test_mode_diagnostic_analytic_gradient_matches_autograd_and_distinguishes_clipping():
    learner = make_learner()
    module = learner.module["uc"]
    batch = learner._convert_batch_type(MultiAgentBatch({"uc": sample_batch(module, 32)}, 32))["uc"]
    output = module.evaluate_actions(batch)
    # Mode ratio is one; the continuous factor alone crosses the joint bound.
    ratio = torch.tensor([1.4, 0.6, 1.4, 0.6] * 8, dtype=torch.float64)
    batch["advantages"] = torch.tensor([1., -1., -1., 1.] * 8)
    batch["action_logp"] = output["action_logp"].detach() - ratio.log()
    batch["logp_parts"] = output["logp_parts"].detach().clone()
    batch["logp_parts"][:, 0] -= ratio.log()
    rng, weights = random_state(), copy.deepcopy(module.get_state())
    diagnostic = mode_statistics(module, batch, output, 0.2)
    assert_nested_equal(rng, random_state())
    assert_nested_equal(weights, module.get_state())
    assert all(p.grad is None for p in module.parameters())
    assert diagnostic["range_clip_fraction"]["joint"] == 1
    assert diagnostic["flat_clip_fraction"]["joint"] == 0.5
    assert diagnostic["mode_blocked_by_joint_fraction"] == 0.5
    assert diagnostic["flat_clip_fraction"]["mode"] == 0
    for label, value, loss in (
        ("joint", ratio, clipped_policy_loss(output["action_logp"], batch["action_logp"], batch["advantages"], .2)[0]),
        ("unclipped", ratio, -(ratio * (output["action_logp"] - output["action_logp"].detach()).exp() * batch["advantages"]).mean()),
        ("separate_mode", torch.ones_like(ratio), clipped_policy_loss(output["logp_parts"][:, 1], batch["logp_parts"][:, 1], batch["advantages"], .2)[0]),
    ):
        grads = torch.autograd.grad(loss, (*module.actor["mode"].parameters(), *module.actor["mode_soc"].parameters()), retain_graph=True)
        norm = torch.cat([g.flatten().double() for g in grads]).norm()
        assert diagnostic["policy_mode_gradient_norm"][label] == pytest.approx(float(norm), rel=2e-6, abs=1e-8)
    json.dumps(diagnostic, allow_nan=False)
    assert flat_surrogate(ratio, batch["advantages"], .2).tolist() == [True, True, False, False] * 8


def test_passive_diagnostics_leave_adam_rng_and_gae_batch_identical(monkeypatch):
    torch.set_num_threads(1)
    learner, reference = make_learner(), make_learner()
    reference.set_state(copy.deepcopy(learner.get_state()))
    data = sample_batch(learner.module["uc"], 64)
    data["obs"][:, 2] = np.linspace(0, 10, 64)
    # Resample once for these synthetic observations before fixing the behavior batch.
    with torch.no_grad():
        for key, value in learner.module["uc"].draw(torch.tensor(data["obs"]), torch.tensor(data["bounds"])).items():
            data[key] = value.numpy()
    data["advantages_raw"] = data["advantages"].copy() * 7 + 2
    batch = MultiAgentBatch({"uc": data}, 64)
    original, rng = copy.deepcopy(batch), random_state()
    learner.update(batch=batch, update_phase="actor", num_epochs=2, minibatch_size=32, shuffle_batch_per_epoch=True)
    final_rng = random_state()
    diagnostics = learner._mode_diagnostics["uc"]
    assert diagnostics["unique_samples"] == 64
    assert sum(row["count"] for row in diagnostics["soc_advantages"]) == 64
    assert diagnostics["summary"]["sample_visits"] == 128
    assert [row["epoch"] for row in diagnostics["minibatches"]] == [1, 1, 2, 2]
    assert [row["adam_step_after"] for row in diagnostics["minibatches"]] == [1, 2, 3, 4]
    assert [row["adam_step_before"] for row in diagnostics["minibatches"]] == [0, 1, 2, 3]
    assert any(row["parameter_delta_norm"]["mode_soc.weight"] > 0 for row in diagnostics["minibatches"])
    assert all(row["gradient_norm_before"]["mode_soc"] is not None for row in diagnostics["minibatches"])
    json.dumps(diagnostics, allow_nan=False)
    # Suppress only the observation hook; retain the identical production loss and native Adam path.
    compute = reference.compute_loss_for_module
    def no_diagnostics(**kwargs):
        reference._mode_diagnostics.clear()
        return compute(**kwargs)
    monkeypatch.setattr(reference, "compute_loss_for_module", no_diagnostics)
    restore_random_state(rng)
    reference.update(batch=copy.deepcopy(original), update_phase="actor", num_epochs=2,
                     minibatch_size=32, shuffle_batch_per_epoch=True)
    assert_nested_equal(final_rng, random_state())
    assert_nested_equal(learner.get_state(), reference.get_state())
    assert_nested_equal(dict(original["uc"]), dict(batch["uc"]))
