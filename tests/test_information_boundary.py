import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from raymarl import IPPOConfig, MAACConfig, MAPPOConfig
from raymarl.distributions import HybridDistribution
from raymarl.learner import JointTorchLearner
from raymarl.networks import ActorModule, JointModule
from raymarl.ppo import PPOModule
from raymarl.spaces import ActionLayout, ObservationLayout


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


@pytest.mark.cuda
def test_cuda_history_hybrid_update():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    o = gym.spaces.Dict(
        {
            "current": gym.spaces.Box(-1, 1, (3,), np.float32),
            "history": gym.spaces.Box(-1, 1, (4, 2), np.float32),
            "history_mask": gym.spaces.Box(0, 1, (4,), np.float32),
        }
    )
    a = gym.spaces.Dict({"box": gym.spaces.Box(-1, 1, (1,), np.float32), "binary": gym.spaces.MultiBinary(1)})
    cfg = (
        MAACConfig()
        .learners(num_learners=0, num_gpus_per_learner=1)
        .training(hidden_dim=8, baseline_samples=2)
    )
    module = JointModule([o, o], [a, a], 8)
    learner = JointTorchLearner(config=cfg, module=module)
    learner.build()
    raw = {
        "current": np.zeros(3, np.float32),
        "history": np.zeros((4, 2), np.float32),
        "history_mask": np.ones(4, np.float32),
    }
    obs = [ObservationLayout(o).tensor([raw] * 4, "cuda")] * 2
    actions = [torch.zeros(4, ActionLayout(a).feature_dim, device="cuda")] * 2
    result = learner.update_joint(
        obs, actions, torch.ones(4, 2, device="cuda"), obs, torch.zeros(4, 2, device="cuda")
    )
    assert np.isfinite(result["critic_loss"])
    assert next(learner.module.actors[0].parameters()).is_cuda
