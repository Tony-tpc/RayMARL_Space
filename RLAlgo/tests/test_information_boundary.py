import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.columns import Columns

from raymarl import IPPOConfig, MAPPOConfig
from raymarl.common.models import ActorModule
from raymarl.distributions import HybridDistribution
from raymarl.ppo import PPOModule


def test_mappo_actor_cannot_use_central_state():
    local = gym.spaces.Box(-1, 1, (3,), np.float32)
    space = gym.spaces.Dict({"local": local, "critic": gym.spaces.Box(-1, 1, (6,), np.float32)})
    actor = PPOModule(
        observation_space=space,
        action_space=gym.spaces.Box(-1, 1, (1,), np.float32),
        model_config={"hidden_dim": 8},
    )
    one = {Columns.OBS: {"local": torch.zeros(2, 3), "critic": torch.zeros(2, 6)}}
    two = {Columns.OBS: {"local": torch.zeros(2, 3), "critic": torch.ones(2, 6)}}
    assert torch.equal(
        actor.forward_inference(one)[Columns.ACTION_DIST_INPUTS],
        actor.forward_inference(two)[Columns.ACTION_DIST_INPUTS],
    )
    assert not torch.equal(actor.compute_values(one), actor.compute_values(two))


def test_actor_standard_rlmodule_interface():
    actor = ActorModule(
        observation_space=gym.spaces.Box(-1, 1, (3,), np.float32),
        action_space=gym.spaces.Discrete(3),
        model_config={"hidden_dim": 8},
    )
    out = actor.forward_inference({Columns.OBS: torch.zeros(2, 3)})
    assert out[Columns.ACTION_DIST_INPUTS].shape == (2, 3)
    assert actor.get_train_action_dist_cls().required_input_dim(actor.action_space) == 3


def test_ppo_density_preserves_saturated_pre_tanh_sample():
    space = gym.spaces.Box(-1, 1, (1,), np.float32)
    old = HybridDistribution(torch.tensor([[20.0, -20.0]]), action_space=space)
    features, logp, _, latent = old.draw(return_latents=True)
    action = old.layout.decode_features(features)
    replay = HybridDistribution(
        torch.tensor([[20.0, -20.0]], requires_grad=True), action_space=space, action_latents=latent
    )
    actual = replay.logp(action)
    assert torch.equal(actual, logp)
    assert torch.isfinite(actual).all()


def test_ppo_does_not_double_transform_physical_actions():
    assert IPPOConfig().normalize_actions is False
    assert MAPPOConfig().normalize_actions is False
