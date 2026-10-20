"""Probability, information-boundary and native optimization checks."""

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from snmarket.action import validate_consumer_action
from snmarket.contracts import LocalBounds

from raymarl.algorithms.stackelberg.learner import StackelbergTorchLearner, clipped_policy_loss
from raymarl.algorithms.stackelberg.rl_module import BOUNDS_FIELDS, StackelbergRLModule


def module_spec(role, inference_only=False):
    return RLModuleSpec(
        module_class=StackelbergRLModule,
        observation_space=gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
        action_space=gym.spaces.Box(-np.inf, np.inf, (6 if role == "uc" else 4,), np.float64),
        inference_only=inference_only,
        model_config={"role": role, "hidden_dim": 12, "critic_dim": 13},
    )


def test_critic_layernorm_and_zero_time_branch_preserve_actor_initialization(monkeypatch):
    from raymarl.algorithms.stackelberg import rl_module as implementation

    specs = [module_spec("uc"), module_spec("consumer")]
    specs[0].model_config["uc_time_features"] = {"elapsed": 0, "remaining": 1, "cycle_steps": 48}
    torch.manual_seed(42)
    current = [spec.build() for spec in specs]
    rng = torch.get_rng_state().clone()
    with monkeypatch.context() as patch:
        patch.setattr(implementation, "ValueNetwork", lambda d, h, t: implementation.mlp(d, h, 1))
        torch.manual_seed(42)
        previous = [spec.build() for spec in specs]
    assert torch.equal(rng, torch.get_rng_state())
    for new, old in zip(current, previous, strict=True):
        for name, value in new.actor.state_dict().items():
            torch.testing.assert_close(value, old.actor.state_dict()[name], rtol=0, atol=0)
        assert sum(isinstance(layer, torch.nn.LayerNorm) for layer in new.value.encoder) == 2
    uc = current[0]
    obs = torch.zeros(2, 7)
    obs[:, 0], obs[:, 1] = torch.tensor([0., 25.]), torch.tensor([100., 75.])
    before = uc.compute_values({"obs": obs}).detach()
    with torch.no_grad():
        uc.value.time_weight[0] = 2
    torch.testing.assert_close(uc.compute_values({"obs": obs}) - before, torch.tensor([0., 0.5]))
    assert "value.time_weight" in dict(uc.named_parameters())
    assert current[1].value.time_weight is None


def legal_bounds(*, zero=False, due=False):
    return LocalBounds(
        cut_max_kw=0.0 if zero else 2.0,
        due_kw=0.0 if zero else (1.0 if due else 0.0),
        due_kwh=0.25 if due else 0.0,
        backlog_kwh=0.0 if zero else 1.0,
        backlog_limit_kwh=2.0,
        baseline_kw=0.0 if zero else 5.0,
        rigid_kw=0.0 if zero else 2.0,
        flex_kw=0.0 if zero else 3.0,
        dt_hours=0.25,
        allow_new_shift=not zero,
        der_bid_min=0.2,
        der_bid_max=0.2 if zero else 0.8,
        uc_tariff=0.8,
    )


def bounds_tensor(bounds, count):
    return torch.tensor([[getattr(bounds, name) for name in BOUNDS_FIELDS]] * count, dtype=torch.float64)


def sample_batch(module, count=16):
    obs, critic = torch.randn(count, 7), torch.randn(count, 13)
    bounds = bounds_tensor(legal_bounds(), count)
    with torch.no_grad():
        data = {"obs": obs, "critic_obs": critic, "bounds": bounds, **module.draw(obs, bounds)}
        data["vf_preds"] = module.compute_values(data)
        data["advantages"] = torch.linspace(-1.5, 2.0, count)
        data["value_targets"] = data["vf_preds"] + torch.linspace(-0.3, 0.8, count)
    return SampleBatch({key: value.detach().cpu().numpy() for key, value in data.items()})


def make_learner():
    config = AlgorithmConfig().learners(num_learners=0, num_gpus_per_learner=0)
    config.actor_lr, config.critic_lr = 0.001, 0.002
    config.clip_param, config.vf_loss_coeff, config.entropy_coeff, config.grad_clip = 0.2, 0.5, 0.0, 0.5
    learner = StackelbergTorchLearner(
        config=config,
        module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "uc": module_spec("uc"),
                "consumer": module_spec("consumer"),
            }
        ),
    )
    learner.build()
    return learner


def test_uc_uses_raw_unclipped_joint_probability():
    module = module_spec("uc").build()
    obs = torch.randn(3, 7)
    raw = torch.tensor([[-2.0, 1.2, 3.0, 0.0, -2.5, 0.4]] * 3)
    distribution = module._normal(module._features(obs))
    actual = module.evaluate_actions({"obs": obs, "raw_latents": raw})["action_logp"]
    torch.testing.assert_close(actual, distribution.log_prob(raw).sum(-1))
    assert not torch.allclose(actual, distribution.log_prob(raw.clamp(0, 1)).sum(-1))
    draw = module.draw(obs)
    assert not draw["zero_width_mask"].any()
    torch.testing.assert_close(
        draw["action_logp"], module.evaluate_actions({"obs": obs, **draw})["action_logp"], rtol=0, atol=0
    )


@pytest.mark.parametrize("zero,due", [(False, False), (False, True), (True, False)])
def test_consumer_domain_logp_and_zero_width_derivatives(zero, due):
    torch.manual_seed(7)
    module = module_spec("consumer").build()
    count, bounds = 512, legal_bounds(zero=zero, due=due)
    obs, tensor = torch.randn(count, 7), bounds_tensor(bounds, count)
    draw = module.draw(obs, tensor)
    for action in draw["actions"].detach().numpy():
        validate_consumer_action(action, bounds)
    new = module.evaluate_actions({"obs": obs, **draw})
    torch.testing.assert_close(draw["action_logp"], new["action_logp"], rtol=0, atol=0)
    assert draw["actions"].dtype == torch.float64
    if zero:
        assert draw["zero_width_mask"][0].tolist() == [True, True, False, True]
        assert torch.all(draw["raw_latents"][:, [0, 1, 3]] == 0)
    (-new["action_logp"].mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())


def test_physical_jacobian_and_teacher_forcing():
    module = module_spec("consumer").build()
    obs = torch.randn(5, 7)
    draw = module.draw(obs, bounds_tensor(legal_bounds(), 5))
    first = module.evaluate_actions({"obs": obs, **draw})["action_logp"]
    expanded = draw["conditional_bounds"].clone()
    expanded[:, :, 1] = expanded[:, :, 0] + 2 * (expanded[:, :, 1] - expanded[:, :, 0])
    second = module.evaluate_actions({"obs": obs, **draw, "conditional_bounds": expanded})["action_logp"]
    torch.testing.assert_close(second - first, torch.full_like(first, -4 * np.log(2)), atol=1e-12, rtol=0)
    # Changing the stored prefix changes later distributions; evaluate_actions
    # conditions on the sampled prefix instead of resampling it during PPO.
    controls = draw["raw_latents"].clone()
    controls[:, 0] = 0.15
    left = module._consumer_distributions(module._features(obs), controls)[1]
    controls[:, 0] = 0.85
    right = module._consumer_distributions(module._features(obs), controls)[1]
    assert not torch.allclose(left.mean, right.mean)


def test_actor_value_information_and_gradient_boundaries():
    obs, first, second = torch.randn(4, 7), torch.zeros(4, 13), torch.ones(4, 13)
    uc, consumer = module_spec("uc").build(), module_spec("consumer").build()
    torch.testing.assert_close(
        uc.compute_values({"obs": obs, "critic_obs": first}),
        uc.compute_values({"obs": obs, "critic_obs": second}),
    )
    assert not torch.allclose(
        consumer.compute_values({"obs": obs, "critic_obs": first}),
        consumer.compute_values({"obs": obs, "critic_obs": second}),
    )
    before = consumer.draw(obs, bounds_tensor(legal_bounds(), 4), deterministic=True)["actions"]
    consumer.compute_values({"obs": obs, "critic_obs": first}).sum().backward()
    assert all(p.grad is None for p in consumer.actor.parameters())
    after = consumer.draw(obs, bounds_tensor(legal_bounds(), 4), deterministic=True)["actions"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    inference = module_spec("consumer", inference_only=True).build()
    assert not hasattr(inference, "value")
    assert any(key.startswith("value.") for key in consumer.get_state(inference_only=True))


def test_clipped_surrogate_matches_hand_calculation():
    ratio = torch.tensor([1.5, 0.5, 1.5, 0.5], dtype=torch.float64)
    advantage = torch.tensor([2.0, 2.0, -2.0, -2.0], dtype=torch.float64)
    loss, actual = clipped_policy_loss(ratio.log(), torch.zeros(4), advantage, 0.2)
    # min(3,2.4), min(1,1.6), min(-3,-2.4), min(-1,-1.6).
    assert loss.item() == pytest.approx(-(2.4 + 1 - 3 - 1.6) / 4)
    torch.testing.assert_close(actual, ratio)


def test_native_optimizer_freezing_precision_and_older_snapshot_restore():
    learner = make_learner()
    groups = [
        {id(p) for g in optimizer.param_groups for p in g["params"]}
        for optimizer in learner._named_optimizers.values()
    ]
    assert all(not a.intersection(b) for i, a in enumerate(groups) for b in groups[i + 1 :])
    batch = MultiAgentBatch({"consumer": sample_batch(learner.module["consumer"])}, env_steps=16)
    converted = learner._convert_batch_type(batch)
    assert converted["consumer"]["conditional_bounds"].dtype == torch.float64
    assert converted["consumer"]["action_logp"].dtype == torch.float64
    initial_uc = copy.deepcopy(learner.module["uc"].get_state())
    initial_consumer = copy.deepcopy(learner.module["consumer"].get_state())
    learner.update(batch=batch, num_epochs=1, minibatch_size=16)
    assert any(
        not np.array_equal(initial_consumer[k], v) for k, v in learner.module["consumer"].get_state().items()
    )
    for key, value in learner.module["uc"].get_state().items():
        np.testing.assert_array_equal(initial_uc[key], value)
    snapshot = copy.deepcopy(learner.get_state())
    learner.update(batch=batch, num_epochs=1, minibatch_size=16)
    expected = copy.deepcopy(learner.module.get_state())
    learner.set_state(snapshot)
    learner.update(batch=batch, num_epochs=1, minibatch_size=16)
    for name, weights in expected.items():
        for key, value in weights.items():
            np.testing.assert_array_equal(value, learner.module[name].get_state()[key])


def test_entropy_gradient_remains_finite_for_deterministic_coordinates():
    module = module_spec("consumer").build()
    batch = {"obs": torch.randn(16, 7), "bounds": bounds_tensor(legal_bounds(zero=True), 16)}
    entropy = module.policy_entropy(batch)
    entropy.mean().backward()
    assert torch.isfinite(entropy).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
