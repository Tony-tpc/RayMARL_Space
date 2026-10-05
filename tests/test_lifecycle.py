import numpy as np
import pytest
import torch

from raymarl import IPPOConfig, ISACConfig, MAACConfig, MADDPGConfig, MAPPOConfig, MASACConfig
from raymarl.inference import ActorBundle

from .fixtures import InterfaceGame


@pytest.mark.integration
@pytest.mark.parametrize("config_cls", [MAACConfig, MASACConfig, MADDPGConfig, ISACConfig])
def test_joint_algorithm_lifecycle(config_cls, ray_runtime, tmp_path):
    config = config_cls().environment(InterfaceGame, env_config={"hybrid": config_cls != MADDPGConfig})
    config.training(
        hidden_dim=8, batch_size=4, warmup_steps=4, baseline_samples=1, storage_path=str(tmp_path / "results")
    )
    algo = config.build_algo()
    try:
        result = algo.train()
        assert result["num_env_steps_sampled_lifetime"] == 4
        assert result["num_updates_lifetime"] == 1
        saved = algo.save_to_path(str(tmp_path / "checkpoint"))
        next_result = algo.train()
        expected = algo.joint_learner.get_state()
        algo.restore_from_path(saved)
        repeated = algo.train()
        assert repeated["num_env_steps_sampled_lifetime"] == next_result["num_env_steps_sampled_lifetime"]
        for aid, state in expected["module"]["actors"].items():
            for key, value in state.items():
                assert torch.equal(value, algo.joint_learner.module.get_state()["actors"][aid][key])
        actor_path = tmp_path / "actors.pt"
        algo.export_actors(actor_path)
        bundle = ActorBundle(actor_path)
        observations, _ = algo.joint_env.reset(seed=10)
        actual = bundle.actions(observations)
        for aid, action in actual.items():
            assert algo.joint_env.get_action_space(aid).contains(action)
    finally:
        algo.stop()


@pytest.mark.integration
@pytest.mark.parametrize("config_cls", [IPPOConfig, MAPPOConfig])
def test_ppo_config_reuse(config_cls, ray_runtime, tmp_path):
    config = (
        config_cls()
        .environment(InterfaceGame, env_config={"hybrid": True})
        .multi_agent(policies={"a", "b"}, policy_mapping_fn=lambda a, *args, **kw: a)
        .env_runners(num_env_runners=0, rollout_fragment_length=16)
        .learners(num_learners=0, num_gpus_per_learner=0)
        .rl_module(model_config={"hidden_dim": 8})
        .training(train_batch_size_per_learner=16, minibatch_size=8, num_epochs=1)
    )
    algo = config.build_algo()
    try:
        result = algo.train()
        assert result["num_env_steps_sampled_lifetime"] >= 16
        assert np.isfinite(result["env_runners"]["episode_return_mean"])
        algo.save_to_path(str(tmp_path / "ppo-checkpoint"))
    finally:
        algo.stop()
