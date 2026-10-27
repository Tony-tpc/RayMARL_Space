"""Numerical and state tests; no environment rollout or pretraining."""
import copy

import numpy as np
import pytest
import torch

from raymarl.algorithms.stackelberg.learner import value_fit_statistics
from raymarl.algorithms.stackelberg.popart import POPART, register_popart, update_popart
from raymarl.algorithms.stackelberg.rl_module import ValueNetwork


def model():
    torch.manual_seed(17)
    value = ValueNetwork(5, 16)
    register_popart(value, 20.0)
    return value


def raw(value, x):
    return value(x) * value.popart_scale + value.popart_shift


def test_raw_values_preserved_across_multiple_target_scales():
    value = model()
    x = torch.randn(64, 5)
    expected = raw(value, x).detach()
    identities = [id(p) for p in value.parameters()]
    for targets in (torch.linspace(-926, 172, 1536), torch.linspace(50, 350, 1536), torch.zeros(1536)):
        update_popart(value, targets)
        torch.testing.assert_close(raw(value, x), expected, atol=5e-5, rtol=2e-5)
        assert [id(p) for p in value.parameters()] == identities
    assert value.popart_updates.item() == 3


def test_statistics_match_independent_debiased_ema():
    value = model()
    batches = [np.array([-20., 10., 100.]), np.array([3., 4., 15.])]
    beta = POPART["beta_per_phase"]
    mean = second = mass = 0.
    for batch in batches:
        mean, second, mass = beta*mean+(1-beta)*batch.mean(), beta*second+(1-beta)*(batch**2).mean(), beta*mass+1-beta
        update_popart(value, batch)
    assert float(value.popart_shift) == pytest.approx(mean/mass, rel=1e-6)
    assert float(value.popart_scale) == pytest.approx(max(second/mass-(mean/mass)**2, .01)**.5, rel=1e-6)


def test_loss_is_normalized_mse_but_metrics_remain_raw():
    value = model()
    x, target = torch.randn(32, 5), torch.linspace(-50, 200, 32)
    update_popart(value, target)
    prediction = raw(value, x)
    loss, rmse, _ = value_fit_statistics(prediction, target, float(value.popart_scale))
    normalized_target = (target-value.popart_shift)/value.popart_scale
    torch.testing.assert_close(loss, .5*(value(x)-normalized_target).square().mean())
    torch.testing.assert_close(rmse, (prediction-target).square().mean().sqrt())
    loss.backward()
    assert value.head.weight.grad is not None
    assert not value.popart_scale.requires_grad


def test_state_restore_and_adam_identity():
    value = model()
    optimizer = torch.optim.Adam(value.parameters(), lr=.001)
    x = torch.randn(32, 5)
    raw(value, x).square().mean().backward()
    optimizer.step()
    saved = copy.deepcopy(value.state_dict())
    moments = copy.deepcopy(optimizer.state_dict())
    update_popart(value, torch.linspace(-900, 200, 1536))
    for key, state in moments["state"].items():
        for field, expected in state.items():
            torch.testing.assert_close(optimizer.state_dict()["state"][key][field], expected, atol=0, rtol=0)
    value.load_state_dict(saved)
    assert all(torch.equal(v, saved[k]) for k, v in value.state_dict().items())


def test_constant_target_has_finite_positive_scale():
    value = model()
    x = torch.randn(4, 5)
    before = raw(value, x).detach()
    update_popart(value, torch.full((1536,), 200.))
    assert float(value.popart_scale) == pytest.approx(.1)
    torch.testing.assert_close(raw(value, x), before, atol=5e-5, rtol=2e-5)


def test_invalid_targets_do_not_modify_state():
    value = model()
    saved = copy.deepcopy(value.state_dict())
    with pytest.raises(ValueError):
        update_popart(value, torch.tensor([float("nan")]))
    assert all(torch.equal(v, saved[k]) for k, v in value.state_dict().items())


def test_native_module_restore_keeps_float64_ema_exact():
    from .test_stackelberg_learner_phases import specification

    module = specification("uc", online_popart=True, value_scale=20.0).build()
    update_popart(module.value, torch.linspace(-926, 172, 1536))
    saved = copy.deepcopy(module.get_state())
    update_popart(module.value, torch.linspace(0, 900, 1536))
    module.set_state(saved)
    for key, value in saved.items():
        np.testing.assert_array_equal(module.get_state()[key], value)


def test_native_learner_updates_statistics_once_and_actor_leaves_values_frozen():
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.policy.sample_batch import MultiAgentBatch

    from raymarl.algorithms.stackelberg.learner import StackelbergTorchLearner

    from .test_stackelberg_learner_phases import assert_nested_equal, specification
    from .test_stackelberg_policy import make_learner, sample_batch

    torch.set_num_threads(1)
    cfg = make_learner().config
    spec = MultiRLModuleSpec(rl_module_specs={
        key: specification(key, online_popart=True) for key in ("uc", "consumer")})
    learner = StackelbergTorchLearner(config=cfg, module_spec=spec)
    learner.build()
    batch = MultiAgentBatch({key: sample_batch(learner.module[key], 64)
                            for key in ("uc", "consumer")}, 64)
    actor = {key: copy.deepcopy(module.actor.state_dict()) for key, module in learner.module.items()}
    learner.update(batch=batch, update_phase="critic", num_epochs=2, minibatch_size=32)
    for key, module in learner.module.items():
        assert module.value.popart_updates.item() == 1
        assert_nested_equal(actor[key], module.actor.state_dict())
    saved = copy.deepcopy(learner.get_state())
    values = {key: copy.deepcopy(m.value.state_dict()) for key, m in learner.module.items()}
    learner.update(batch=batch, update_phase="actor", num_epochs=1, minibatch_size=32)
    for key, module in learner.module.items():
        assert_nested_equal(values[key], module.value.state_dict())
    learner.set_state(saved)
    assert_nested_equal(saved, learner.get_state())


@pytest.mark.integration
def test_popart_always_round_restores_consumers_and_checkpoint(ray_runtime, tmp_path):
    from experiments.sn_energy_market.runtime import digest_tree
    from experiments.sn_energy_market.training_trace import TrainingTrace

    from .test_stackelberg_resident import mixed_config

    torch.set_num_threads(1)
    cfg = mixed_config(tmp_path, 4).training(always_keep_candidate=True)
    cfg.rl_module(model_config={"hidden_dim": 128, "online_popart": True})
    algorithm = cfg.build_algo()
    trace = TrainingTrace(algorithm, tmp_path, write_trajectory=False).install()
    try:
        algorithm.train()
        trace.assert_budget(1)
        assert algorithm.last_round["selected_candidate"] == 1
        for key, module in algorithm.learner_group._learner.module.items():
            assert module.value.popart_updates.item() == (3 if key == "uc" else 2)
        trace.close()
        checkpoint = algorithm.save_to_path()
        saved = algorithm.get_state()
        first = algorithm.env_runner.collect_phase(16)
        algorithm.restore_from_path(checkpoint)
        assert digest_tree(first.policy_batches) == digest_tree(algorithm.env_runner.collect_phase(16).policy_batches)
        algorithm.set_state(saved)
        assert digest_tree(saved) == digest_tree(algorithm.get_state())
    finally:
        trace.close()
        algorithm.stop()


