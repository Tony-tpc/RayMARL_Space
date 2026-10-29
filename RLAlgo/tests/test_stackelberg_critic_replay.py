"""Replay ownership, fresh PopArt, and candidate/checkpoint reproducibility."""

import copy

import numpy as np
import pytest
import torch
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

from raymarl.algorithms.stackelberg.critic_replay import CriticReplay
from raymarl.algorithms.stackelberg.learner import StackelbergTorchLearner
from raymarl.algorithms.stackelberg.popart import update_popart
from raymarl.algorithms.stackelberg.runner import random_state

from .test_stackelberg_algorithm import configuration
from .test_stackelberg_learner_phases import assert_nested_equal, specification
from .test_stackelberg_policy import make_learner, sample_batch


def test_replay_bounds_no_mutation_private_rng_and_exact_restore():
    pool = CriticReplay(12, 11)
    outside = random_state()
    for index in range(15):
        data = SampleBatch({"obs": np.arange(32)[:, None] + 32 * index,
                            "value_targets": np.arange(32) + 32 * index})
        original = copy.deepcopy(dict(data))
        state = pool.get_state()
        first = pool.mix(data)
        pool.set_state(state)
        second = pool.mix(data)
        assert_nested_equal(dict(first), dict(second))
        assert_nested_equal(dict(data), original)
        assert len(pool.entries) == min(index + 1, 12)
        assert len(np.unique(first["obs"])) == 32
        assert_nested_equal(first["obs"][:, 0], first["value_targets"])
        assert first["obs"].min() >= max(0, index - 11) * 32
    assert_nested_equal(outside, random_state())


def test_replay_uses_only_fresh_targets_for_popart_and_keeps_actor_frozen():
    torch.set_num_threads(1)
    learner = StackelbergTorchLearner(config=make_learner().config,
        module_spec=MultiRLModuleSpec(rl_module_specs={
            "uc": specification("uc", online_popart=True)}))
    learner.build()
    module = learner.module["uc"]
    actor = copy.deepcopy(module.actor.state_dict())
    data = sample_batch(module, 64)
    data["value_targets"][:] = -1000.0  # Old replay targets differ strongly from new observations.
    fresh = np.linspace(20., 80., 64)
    expected = copy.deepcopy(module.value)
    update_popart(expected, fresh)
    learner.update(batch=MultiAgentBatch({"uc": data}, 64), update_phase="critic",
                   popart_targets={"uc": fresh}, num_epochs=2, minibatch_size=32)
    for name, value in expected.named_buffers():
        torch.testing.assert_close(dict(module.value.named_buffers())[name], value, atol=0, rtol=0)
    assert_nested_equal(actor, module.actor.state_dict())
    assert module.value.popart_updates == 1


@pytest.mark.integration
def test_replay_candidate_and_disk_checkpoint_restore(ray_runtime, tmp_path):
    torch.set_num_threads(1)
    cfg = configuration(tmp_path).training(uc_critic_replay_batches=12, deviation_updates=0)
    algo = cfg.build_algo()
    try:
        algo._adapt(algo.consumer_ids, 1)
        saved = algo._snapshot()
        algo._adapt(("uc",), 1)
        expected = algo._snapshot()
        algo._restore(saved)
        algo._adapt(("uc",), 1)
        actual = algo._snapshot()
        assert_nested_equal(expected["learner"], actual["learner"])
        assert_nested_equal(expected["uc_critic_replay"], actual["uc_critic_replay"])
        checkpoint = algo.save_to_path(str(tmp_path / "checkpoint"))
        algo._adapt(algo.consumer_ids, 1)
        algo.restore_from_path(checkpoint)
        assert_nested_equal(actual["uc_critic_replay"], algo.uc_critic_replay.get_state())
    finally:
        algo.stop()
