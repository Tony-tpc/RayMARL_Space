import copy

import numpy as np
import pytest
import torch

from raymarl import ISACConfig, MAACConfig, MADDPGConfig, MASACConfig
from raymarl.common.connectors import CRITIC_ID
from raymarl.common.models import AttentionCritic, TwinJointQCritic, TwinLocalQCritic
from raymarl.common.replay import JointReplay

from .native_helpers import make_learner

CONFIGS = [MAACConfig, MASACConfig, MADDPGConfig, ISACConfig]


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_module_learning_rates_survive_native_scheduler_update(config_cls):
    learner, episodes = make_learner(
        config_cls, hybrid=config_cls != MADDPGConfig,
        module_overrides={
            "a": {"actor_lr": 0.0123, "critic_lr": 0.0456},
            CRITIC_ID: {"critic_lr": 0.0678},
        },
    )
    for updated in (False, True):
        if updated:
            learner.update(episodes=episodes)
        for name, optimizer in learner._named_optimizers.items():
            module_id = learner._optimizer_name_to_module[name]
            role = "actor" if name.endswith("actor") else "critic"
            expected = (
                (0.0123 if role == "actor" else 0.0456) if module_id == "a"
                else 0.0678 if module_id == CRITIC_ID
                else 0.0003 if role == "actor" else 0.001
            )
            assert all(group["lr"] == pytest.approx(expected) for group in optimizer.param_groups)


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_native_update_optimizer_ownership_and_restore(config_cls):
    learner, episodes = make_learner(config_cls, hybrid=config_cls != MADDPGConfig)
    groups = [
        {id(p) for group in opt.param_groups for p in group["params"]}
        for opt in learner._named_optimizers.values()
    ]
    for i, group in enumerate(groups):
        assert group
        for other in groups[i + 1 :]:
            assert not group & other
    assert "central_critic" not in learner.module.get_state(inference_only=True)
    learner.update(episodes=episodes)
    state = learner.get_state()
    assert learner.steps == 1
    learner.update(episodes=episodes)
    expected = copy.deepcopy(learner.module.get_state())
    learner.set_state(state)
    learner.update(episodes=episodes)
    for mid, weights in expected.items():
        for key, value in weights.items():
            np.testing.assert_array_equal(value, learner.module.get_state()[mid][key])


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_nonfinite_checkpoint_rejected_before_weight_changes(config_cls):
    learner, _ = make_learner(config_cls, hybrid=config_cls != MADDPGConfig)
    before = copy.deepcopy(learner.module.get_state())
    state = learner.get_state()
    mid = next(iter(state["rl_module"]))
    key = next(iter(state["rl_module"][mid]))
    state["rl_module"][mid][key].flat[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        learner.set_state(state)
    for mid, weights in before.items():
        for key, value in weights.items():
            np.testing.assert_array_equal(value, learner.module.get_state()[mid][key])


def test_distinct_critic_definitions_and_target_policies():
    maac, _ = make_learner(MAACConfig)
    masac, _ = make_learner(MASACConfig)
    isac, _ = make_learner(ISACConfig)
    assert isinstance(maac.module[CRITIC_ID].critic, AttentionCritic)
    assert isinstance(masac.module[CRITIC_ID].critic, TwinJointQCritic)
    for mid in ("a", "b"):
        assert hasattr(maac.module[mid], "target_policy")
        assert not hasattr(masac.module[mid], "target_policy")
        assert isinstance(isac.module[mid].critic, TwinLocalQCritic)
        assert not hasattr(isac.module[mid], "target_policy")
    assert not hasattr(isac.config, "counterfactual")


def test_replay_episode_alignment_and_fragment_time():
    learner, episodes = make_learner(MAACConfig)
    replay = JointReplay(8, 7)
    replay.add_episodes(episodes, ("a", "b"))
    assert len(replay) == 4
    sampled = replay.sample_episodes(
        6, learner.config.policy_mapping_fn, episodes[0].observation_space, episodes[0].action_space
    )
    assert all(len(e) == 1 for e in sampled)
    assert all(set(e.get_actions(0)) == {"a", "b"} for e in sampled)
    bad = copy.deepcopy(episodes[0])
    bad.agent_episodes["b"].is_terminated = False
    with pytest.raises(ValueError, match="Partial"):
        replay.add_episodes([bad], ("a", "b"))


@pytest.mark.parametrize("config_cls", [MAACConfig, MASACConfig, ISACConfig])
def test_history_and_hybrid_backward_cpu(config_cls):
    learner, episodes = make_learner(config_cls, history=True)
    learner.update(episodes=episodes)
    assert learner.steps == 1
    assert all(
        np.isfinite(value).all()
        for weights in learner.module.get_state().values()
        for value in weights.values()
    )


@pytest.mark.cuda
def test_cuda_history_hybrid_backward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    learner, episodes = make_learner(MAACConfig, history=True, gpu=1)
    learner.update(episodes=episodes)
    assert learner.steps == 1
    assert next(learner.module["a"].parameters()).is_cuda
