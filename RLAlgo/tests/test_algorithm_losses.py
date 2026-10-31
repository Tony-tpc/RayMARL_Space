"""Numerical checks for actual per-algorithm losses and data boundaries."""

import copy

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from torch import nn

from raymarl import ISACConfig, MAACConfig, MADDPGConfig, MASACConfig
from raymarl.common.connectors import CRITIC_ID, JOINT_ROW
from raymarl.common.models import Encoder

from .fixtures import InterfaceGame
from .native_helpers import make_episodes, make_learner


def training_batch(learner, episodes):
    return learner._learner_connector(
        rl_module=learner.module, batch={}, episodes=copy.deepcopy(episodes), shared_data={}
    )


def constant_local(network, value):
    with torch.no_grad():
        for p in network.parameters():
            p.zero_()
        network.head[-1].bias.fill_(value)


def constant_joint(network, value):
    with torch.no_grad():
        for p in network.parameters():
            p.zero_()
        for head in network.heads:
            head[-1].bias.fill_(value)


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
@pytest.mark.parametrize("config_cls", [ISACConfig, MASACConfig])
def test_twin_q_target_uses_minimum_and_only_termination_stops_bootstrap(config_cls, terminated, truncated):
    learner, _ = make_learner(config_cls)
    learner.config.temperature = 0.0
    env = InterfaceGame({"hybrid": True})
    episodes = make_episodes(
        env, learner.config.policy_mapping_fn, terminated=terminated, truncated=truncated
    )
    batch = training_batch(learner, episodes)
    if config_cls == ISACConfig:
        module = learner.module["a"]
        constant_local(module.critic.q1, 0)
        constant_local(module.critic.q2, 0)
        constant_local(module.target_critic.q1, 2)
        constant_local(module.target_critic.q2, 5)
        loss = learner.compute_loss_for_module(
            module_id="a", config=learner.config, batch=batch["a"], fwd_out={}
        )
        expected = batch["a"][Columns.REWARDS] + (0 if terminated else 2 * learner.config.gamma)
        assert loss.item() == pytest.approx((2 * expected.square().mean()).item(), rel=1e-6)
    else:
        module = learner.module[CRITIC_ID]
        constant_joint(module.critic.q1, 0)
        constant_joint(module.critic.q2, 0)
        constant_joint(module.target_critic.q1, 2)
        constant_joint(module.target_critic.q2, 5)
        loss = learner.compute_losses(batch=batch, fwd_out={})[CRITIC_ID]
        expected = torch.stack([batch[mid][Columns.REWARDS] for mid in ("a", "b")], -1)
        expected = expected + (0 if terminated else 2 * learner.config.gamma)
        assert loss.item() == pytest.approx((2 * expected.square().mean(0).sum()).item(), rel=1e-6)


class BinaryTwin(nn.Module):
    def forward(self, obs, actions):
        q = 3 * actions[:, 0]
        return q, q


def test_isac_exact_binary_policy_gradient_and_local_information():
    learner, episodes = make_learner(ISACConfig)
    module = learner.module["a"]
    module.critic = BinaryTwin()
    learner.config.temperature = 0
    learner._phase = "actor"
    with torch.no_grad():
        module.head.weight.zero_()
        module.head.bias.zero_()
        module.head.bias[0] = 0.4
    batch = training_batch(learner, episodes)
    loss = learner.compute_loss_for_module(module_id="a", config=learner.config, batch=batch["a"], fwd_out={})
    gradient = torch.autograd.grad(loss, module.head.bias)[0]
    p = torch.sigmoid(torch.tensor(0.4))
    assert loss.item() == pytest.approx((-3 * p).item(), rel=1e-6)
    assert gradient[0].item() == pytest.approx((-3 * p * (1 - p)).item(), rel=1e-6)
    assert torch.allclose(gradient[1:], torch.zeros_like(gradient[1:]))
    rng = torch.get_rng_state()
    first = learner.compute_loss_for_module(
        module_id="a", config=learner.config, batch=batch["a"], fwd_out={}
    )
    changed = copy.deepcopy(batch)
    changed["b"][Columns.OBS].fill_(99)
    torch.set_rng_state(rng)
    second = learner.compute_loss_for_module(
        module_id="a", config=learner.config, batch=changed["a"], fwd_out={}
    )
    assert torch.equal(first, second)


def test_maac_baseline_cancels_constant_q_without_entropy():
    learner, episodes = make_learner(MAACConfig)
    constant_joint(learner.module[CRITIC_ID].critic, 3)
    learner.config.temperature = 0
    learner._phase = "actor"
    batch = training_batch(learner, episodes)
    losses = learner.compute_losses(batch=batch, fwd_out={})
    assert all(abs(loss.item()) < 1e-6 for loss in losses.values())


def test_maddpg_other_actor_has_no_gradient_from_this_actor_loss():
    learner, episodes = make_learner(MADDPGConfig, hybrid=False)
    learner._phase = "actor"
    batch = training_batch(learner, episodes)
    losses = learner.compute_losses(batch=batch, fwd_out={})
    gradients = torch.autograd.grad(losses["a"], learner.module["b"].online_parameters(), allow_unused=True)
    assert all(gradient is None for gradient in gradients)


def test_permuted_joint_rows_are_rejected():
    learner, episodes = make_learner(MAACConfig)
    batch = training_batch(learner, episodes)
    batch["b"][JOINT_ROW] = batch["b"][JOINT_ROW].flip(0)
    with pytest.raises(ValueError, match="row identities"):
        learner.compute_losses(batch=batch, fwd_out={})


def test_history_mask_hides_padding_and_empty_history():
    env = InterfaceGame()
    import gymnasium as gym

    space = gym.spaces.Dict(
        {
            "current": env.get_observation_space("a"),
            "history": gym.spaces.Box(-1, 1, (3, 2), np.float32),
            "history_mask": gym.spaces.Box(0, 1, (3,), np.float32),
        }
    )
    encoder = Encoder(space, 8)
    first = {
        "current": torch.zeros(2, 2),
        "history": torch.zeros(2, 3, 2),
        "history_mask": torch.tensor([[1, 0, 0], [0, 0, 0]]),
    }
    other = copy.deepcopy(first)
    other["history"][:, 1:] = 100
    other["history"][1, 0] = 100
    assert torch.equal(encoder(first), encoder(other))


def test_model_configuration_has_one_entry_and_rejects_other_checkpoint_layouts():
    config = MAACConfig().rl_module(model_config={"hidden_dim": 8})
    assert config.model_config["hidden_dim"] == 8
    with pytest.raises(TypeError, match="hidden_dim"):
        config.training(hidden_dim=16)
    maac, _ = make_learner(MAACConfig)
    masac, _ = make_learner(MASACConfig)
    with pytest.raises(ValueError, match="layout mismatch"):
        masac.set_state(maac.get_state())
