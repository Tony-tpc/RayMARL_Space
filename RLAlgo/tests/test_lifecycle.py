from pathlib import Path

import numpy as np
import pytest
import torch

from raymarl import IPPOConfig, ISACConfig, MAACConfig, MADDPGConfig, MAPPOConfig, MASACConfig
from raymarl.common.connectors import CRITIC_ID
from raymarl.inference import ActorBundle

from .fixtures import InterfaceGame

CONFIGS = [MAACConfig, MASACConfig, MADDPGConfig, ISACConfig, MAPPOConfig, IPPOConfig]


def small_config(config_cls, root):
    config = (
        config_cls()
        .environment(InterfaceGame, env_config={"hybrid": config_cls != MADDPGConfig})
        .learners(num_learners=0, num_gpus_per_learner=0)
        .env_runners(num_env_runners=0, rollout_fragment_length=4)
        .rl_module(model_config={"hidden_dim": 8})
        .training(storage_path=str(root), checkpoint_interval=0)
    )
    if config_cls in (MAPPOConfig, IPPOConfig):
        config.training(train_batch_size_per_learner=8, minibatch_size=8, num_epochs=1)
    else:
        config.training(batch_size=4, warmup_steps=4)
        if config_cls == MAACConfig:
            config.training(baseline_samples=1)
    return config


@pytest.mark.integration
@pytest.mark.parametrize("config_cls", CONFIGS)
def test_native_algorithm_train_evaluate_restore_export(config_cls, ray_runtime, tmp_path):
    algo = small_config(config_cls, tmp_path / "results").build_algo()
    restored = None
    try:
        assert algo.env_runner_group is not None
        assert algo.learner_group is not None
        assert not hasattr(algo, "_sampler_actors")
        assert CRITIC_ID not in algo.env_runner.module
        for module in algo.env_runner.module.values():
            assert not hasattr(module, "critic")
            assert not hasattr(module, "value")
        result = algo.train()
        with pytest.raises(ValueError, match="Incompatible"):
            algo.set_state({"agent_order": ("a", "b"), "learner": {}})
        assert result["num_env_steps_sampled_lifetime"] >= 4
        assert np.isfinite(result["env_runners"]["episode_return_mean"])
        saved = algo.save_to_path(str(tmp_path / "checkpoint"))
        expected = algo.learner_group.get_state()["learner"]["rl_module"]
        algo.train()
        algo.restore_from_path(saved)
        actual = algo.learner_group.get_state()["learner"]["rl_module"]
        for mid, values in expected.items():
            for key, value in values.items():
                np.testing.assert_array_equal(value, actual[mid][key])
        evaluation = algo.evaluate()
        assert np.isfinite(evaluation["env_runners"]["episode_return_mean"])
        path = tmp_path / "actors.pt"
        algo.export_actors(path)
        payload = torch.load(path, weights_only=False)
        assert all(key.startswith(("encoder.", "head.")) for w in payload["actors"] for key in w)
        bundle = ActorBundle(path)
        env = InterfaceGame({"hybrid": config_cls != MADDPGConfig})
        obs, _ = env.reset(seed=1)
        for aid, action in bundle.actions(obs).items():
            assert env.get_action_space(aid).contains(action)
        assert all(not hasattr(m, "critic") and not hasattr(m, "value") for m in bundle.actors)
        restored = type(algo).from_checkpoint(saved)
        restored.train()
        assert Path(algo.logdir) != Path(restored.logdir)
    finally:
        if restored:
            restored.stop()
        algo.stop()


@pytest.mark.integration
def test_managed_checkpoint_retention_and_isolated_runs(ray_runtime, tmp_path):
    algo = small_config(MAACConfig, tmp_path).build_algo()
    other = None
    try:
        algo.train()
        manual = algo.save_to_path(str(tmp_path / "manual"))
        manual_inside = algo.save_to_path(str(Path(algo.logdir) / "checkpoints" / "checkpoint-manual"))
        for _ in range(3):
            algo.save_to_path()
        assert Path(manual).exists()
        assert Path(manual_inside).exists()
        assert len(list((Path(algo.logdir) / "checkpoints").glob("checkpoint-*/.raymarl-managed"))) == 2
        assert (Path(algo.logdir) / "metrics.jsonl").is_file()
        assert not (Path(algo.logdir) / "details.jsonl").exists()
        other = small_config(MAACConfig, tmp_path).build_algo()
        assert algo.logdir != other.logdir
    finally:
        if other:
            other.stop()
        algo.stop()


@pytest.mark.integration
def test_fragment_sampling_uses_unique_joint_time_rows(ray_runtime, tmp_path):
    config = (
        small_config(MAACConfig, tmp_path)
        .environment(InterfaceGame, env_config={"hybrid": True, "horizon": 5})
        .env_runners(rollout_fragment_length=2)
    )
    algo = config.build_algo()
    try:
        for _ in range(3):
            algo.train()
        assert algo.sampled_steps == 6
        assert len(algo.replay) == 6
        identities = [(row[-2], row[-1]) for row in algo.replay.rows]
        assert len(set(identities)) == 6
        assert sorted(t for eid, t in identities if eid == identities[0][0]) == list(range(5))
    finally:
        algo.stop()
