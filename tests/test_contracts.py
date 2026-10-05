import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from raymarl import MAACConfig, MADDPGConfig, MASACConfig
from raymarl.distributions import HybridDistribution, bound_distribution
from raymarl.learner import JointTorchLearner, score_function_loss, soft_target, soft_update
from raymarl.networks import AttentionCritic, JointModule
from raymarl.replay import JointReplay
from raymarl.spaces import ActionLayout, ObservationLayout


@pytest.fixture
def hybrid():
    return gym.spaces.Dict(
        {
            "continuous": gym.spaces.Box(-1, 1, (2,), np.float32),
            "binary": gym.spaces.MultiBinary(1),
            "choice": gym.spaces.Discrete(3),
        }
    )


def test_composite_roundtrip(hybrid):
    layout = ActionLayout(hybrid)
    action = {"continuous": np.array([-0.2, 0.8], np.float32), "binary": np.array([1]), "choice": 2}
    reconstructed = layout.decode_features(layout.encode(action))
    assert reconstructed["choice"].item() == 2
    assert torch.allclose(reconstructed["continuous"], torch.tensor(action["continuous"]))
    assert reconstructed["binary"].item() == 1
    assert layout.discrete_count == 6


def test_hybrid_log_prob_and_saturation(hybrid):
    x = torch.zeros(4, ActionLayout(hybrid).param_dim, requires_grad=True)
    distribution = HybridDistribution(x, action_space=hybrid)
    action = distribution.sample()
    lp = distribution.logp(action)
    assert torch.isfinite(lp).all()
    (-lp.mean()).backward()
    assert torch.isfinite(x.grad).all()
    box = gym.spaces.Box(-1, 1, (1,), np.float32)
    extreme = torch.tensor([[20.0, -20.0]], requires_grad=True)
    features, logp, _ = HybridDistribution(extreme, action_space=box).draw()
    assert features.item() == 1.0
    assert torch.isfinite(logp).all()
    (-logp.mean()).backward()
    assert torch.isfinite(extreme.grad).all()


def test_bound_distribution(hybrid):
    d = bound_distribution(hybrid).from_logits(torch.zeros(2, ActionLayout(hybrid).param_dim))
    assert set(d.sample()) == set(hybrid.spaces)
    assert torch.allclose(d.kl(d), torch.zeros(2))


def test_attention_self_mask_and_single_agent():
    obs_space = gym.spaces.Box(-1, 1, (3,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
    for n in (1, 2, 3):
        critic = AttentionCritic([obs_space] * n, [action_space] * n, hidden=8)
        observations = [ObservationLayout(obs_space).tensor([np.zeros(3)] * 4, "cpu") for _ in range(n)]
        q, attention = critic(observations, [torch.zeros(4, 1)] * n, return_attention=True)
        assert q.shape == (4, n)
        assert torch.allclose(attention.diagonal(dim1=-2, dim2=-1), torch.zeros(4, n))
        if n == 1:
            assert attention.sum() == 0
        else:
            assert torch.allclose(attention.sum(-1), torch.ones(4, n))
            if n == 2:
                assert torch.equal(attention[:, 0, 1], torch.ones(4))


def test_soft_target_terminal_and_update_direction():
    target = soft_target(
        torch.tensor([[2.0, 2.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[5.0, 5.0]]),
        torch.tensor([[3.0, 3.0]]),
        0.9,
        0.1,
    )
    assert target[0, 0] == 2
    assert target[0, 1] == pytest.approx(2 + 0.9 * (5 - 0.3))
    online, slow = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
    online.weight.data.fill_(1)
    slow.weight.data.zero_()
    soft_update(slow, online, 0.001)
    assert slow.weight.item() == pytest.approx(0.001)


def test_counterfactual_score_gradient_exact():
    logit = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    p = logit.sigmoid()
    q = torch.tensor([1.0, 3.0], dtype=torch.float64)
    probs = torch.stack((1 - p, p))
    baseline = (probs * q).sum().detach()
    # Enumeration supplies exact expectation of score-function samples.
    loss = (probs.detach() * probs.log() * (baseline - q)).sum()
    grad = torch.autograd.grad(loss, logit, retain_graph=True)[0]
    assert grad.item() == pytest.approx((-2 * p * (1 - p)).item(), abs=1e-12)
    score_function_loss(probs.log(), q, baseline, 0.1).backward()
    assert torch.isfinite(logit.grad)


def test_joint_replay_alignment_and_snapshot():
    replay = JointReplay(2, 42)
    for t in range(3):
        v = {"a": np.array([t]), "b": np.array([t + 100])}
        replay.add(v, v, {"a": 1, "b": 2}, v, {"a": False, "b": True}, {"a": True, "b": False}, 0, t)
    snapshot = replay.get_state()
    first = replay.sample(4)
    replay.set_state(snapshot)
    assert [x[-1] for x in replay.sample(4)] == [x[-1] for x in first]
    snapshot["rows"][0][0]["a"][0] = -999
    assert replay.rows[0][0]["a"][0] != -999
    for row in first:
        assert row[0]["b"][0] - row[0]["a"][0] == 100


@pytest.mark.parametrize("config_cls", [MAACConfig, MASACConfig, MADDPGConfig])
def test_learner_update_checkpoint_and_optimizer_ownership(config_cls):
    torch.manual_seed(12)
    o = gym.spaces.Box(-1, 1, (3,), np.float32)
    a = gym.spaces.Box(-1, 1, (1,), np.float32)
    c = config_cls().training(hidden_dim=8, baseline_samples=2, batch_size=4)
    m = JointModule([o] * 2, [a] * 2, 8, c.critic_mode, twin=c.variant == "masac")
    learner = JointTorchLearner(config=c, module=m)
    learner.build()
    actor_params = [{id(p) for p in actor.parameters()} for actor in m.actors]
    critics = {id(p) for p in learner.critic_parameters}
    assert not actor_params[0] & actor_params[1]
    assert not critics & set.union(*actor_params)
    obs = [ObservationLayout(o).tensor([np.zeros(3)] * 4, "cpu")] * 2
    args = (obs, [torch.zeros(4, 1)] * 2, torch.ones(4, 2), obs, torch.zeros(4, 2))
    learner.update_joint(*args)
    state, rng = learner.get_state(), torch.get_rng_state()
    learner.update_joint(*args)
    expected = copy.deepcopy(m.get_state())
    learner.set_state(state)
    torch.set_rng_state(rng)
    learner.update_joint(*args)
    for key, values in expected["actors"].items():
        for name, tensor in values.items():
            assert torch.equal(tensor, m.get_state()["actors"][key][name])


@pytest.mark.parametrize("key,value", [("actor_lr", float("nan")), ("tau", 2), ("batch_size", 0)])
def test_config_rejects_invalid(key, value):
    with pytest.raises(ValueError):
        MAACConfig().training(**{key: value}).validate()


def test_nonfinite_checkpoint_rejected_before_weight_changes():
    o = gym.spaces.Box(-1, 1, (3,), np.float32)
    a = gym.spaces.Box(-1, 1, (1,), np.float32)
    config = MAACConfig().training(hidden_dim=8)
    module = JointModule([o], [a], 8)
    learner = JointTorchLearner(config=config, module=module)
    learner.build()
    state = learner.get_state()
    before = copy.deepcopy(module.get_state())
    key = next(iter(state["module"]["critic"]))
    state["module"]["critic"][key].fill_(float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        learner.set_state(state)
    for k, v in before["critic"].items():
        assert torch.equal(v, module.get_state()["critic"][k])
