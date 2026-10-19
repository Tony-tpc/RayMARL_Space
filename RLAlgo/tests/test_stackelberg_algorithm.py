"""Candidate statistics, independent updates, restore and native training lifecycle."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from raymarl.algorithms.stackelberg import StackelbergConfig
from raymarl.algorithms.stackelberg.algorithm import Stackelberg, paired_gain, select_candidate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def configuration(path, *, device="cpu"):
    from experiments.sn_energy_market.training_env import MarketEnvFactory
    from snmarket import EnvironmentConfig

    cfg = EnvironmentConfig()
    cfg = replace(
        cfg,
        system=replace(
            cfg.system, num_consumers=2, horizon_steps=6, start_datetime="2026-01-01T17:00:00+08:00"
        ),
        market=replace(cfg.market, shift_deadline_steps=2),
    ).validate()
    return (
        StackelbergConfig()
        .scenarios(MarketEnvFactory(cfg), train_seeds=(2, 3), validation_seeds=(11, 12), test_seeds=(21, 22))
        .rl_module(model_config={"hidden_dim": 8})
        .learners(num_gpus_per_learner=int(device == "cuda"))
        .training(
            rollout_length=4,
            inner_updates=1,
            response_updates=1,
            uc_candidates=1,
            uc_updates_per_candidate=1,
            deviation_updates=1,
            num_epochs=1,
            minibatch_size=4,
            storage_path=str(path),
            checkpoint_interval=0,
        )
    )


def rows(values):
    return [{"seed": index, "returns": {"uc": value}} for index, value in enumerate(values)]


def test_candidate_acceptance_includes_incumbent_and_paired_uncertainty():
    incumbent = rows([1, 3, 8])
    better = rows([3, 5, 10])
    noisy = rows([100, -50, 9])
    selected, gains = select_candidate([incumbent, better, noisy])
    assert selected == 1 and gains[1]["lower"] == 2
    assert select_candidate([incumbent, rows([0, 2, 7])])[0] == 0
    assert select_candidate([rows([0]), rows([10])])[0] == 0
    with pytest.raises(ValueError, match="same ordered"):
        paired_gain(incumbent, list(reversed(incumbent)))


def test_disjoint_scenario_splits_and_continuous_contract(tmp_path):
    cfg = configuration(tmp_path)
    cfg.validation_seeds = (2,)
    with pytest.raises(ValueError, match="disjoint"):
        cfg.validate()


def test_training_and_validation_allow_reserved_test_split(tmp_path):
    cfg = configuration(tmp_path)
    cfg.scenarios(
        cfg.scenario_factory,
        train_seeds=(2, 3),
        validation_seeds=(11, 12),
        test_seeds=(),
    )
    cfg.validate()
    assert cfg.learning_contract()["test_seeds"] == []
    for field in ("train_seeds", "validation_seeds"):
        invalid = cfg.copy(copy_frozen=False)
        setattr(invalid, field, ())
        with pytest.raises(ValueError, match="nonempty"):
            invalid.validate()
    cfg.test_seeds = (21, 21)
    with pytest.raises(ValueError, match="distinct"):
        cfg.validate()


def test_reserved_test_requires_explicit_evaluation_seeds():
    calls = []

    def evaluate(seeds, deterministic):
        calls.append((seeds, deterministic))
        return [
            {
                "seed": seed,
                "returns": {"uc": 1.0},
                "physical_periods": 2,
                "metrics": {},
                "bootstrap_used": False,
            }
            for seed in seeds
        ]

    algorithm = SimpleNamespace(config=SimpleNamespace(test_seeds=()), _evaluate=evaluate)
    with pytest.raises(ValueError, match="explicit nonempty evaluation seeds"):
        Stackelberg.evaluate_pair(algorithm)
    with pytest.raises(ValueError, match="explicit nonempty evaluation seeds"):
        Stackelberg.evaluate_pair(algorithm, ())
    assert calls == []
    result = Stackelberg.evaluate_pair(algorithm, (11, 12))
    assert result["summary"]["scenarios"] == 2
    assert calls == [((11, 12), False)]
    algorithm.config.test_seeds = (21, 22)
    Stackelberg.evaluate_pair(algorithm)
    assert calls[-1] == ((21, 22), False)


def test_learning_contract_tracks_targets_and_scenarios_not_output_path(tmp_path):
    left = configuration(tmp_path / "first")
    right = configuration(tmp_path / "second")
    assert left.learning_contract() == right.learning_contract()
    right.gamma = 0.5
    assert left.learning_contract() != right.learning_contract()
    right.gamma = left.gamma
    right.train_seeds = (4, 5)
    assert left.learning_contract() != right.learning_contract()


def assert_nested_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, torch.as_tensor(right), rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


@pytest.mark.integration
def test_native_round_freeze_deviation_and_exact_resume(ray_runtime, tmp_path):
    torch.set_num_threads(1)
    cwd = Path.cwd()
    algo = configuration(tmp_path).build_algo()
    try:
        assert Path.cwd() == cwd
        assert Path(algo.logdir).is_relative_to(tmp_path)
        uc_before = copy.deepcopy(algo.learner_group.get_weights()["uc"])
        consumers_before = copy.deepcopy(algo.learner_group.get_weights()[algo.consumer_ids[0]])
        algo._adapt(algo.consumer_ids, 1, seed=2)
        assert_nested_equal(uc_before, algo.learner_group.get_weights()["uc"])
        assert any(
            not np.array_equal(consumers_before[key], value)
            for key, value in algo.learner_group.get_weights()[algo.consumer_ids[0]].items()
        )
        reference = algo._snapshot()
        deviations = algo.search_deviations()
        assert deviations["searched"] and len(deviations["per_consumer"]) == 2
        assert_nested_equal(reference["learner"], algo._snapshot()["learner"])

        result = algo.train()
        assert result["training_iteration"] == 1
        assert len(algo.last_round["candidates"]) == 2
        assert algo.sampled_periods >= 7 * 4
        assert algo.evaluate_pair()["summary"]["scenarios"] == 2
        snapshot = algo.get_state()
        expected = algo.env_runner.collect(3)
        algo.set_state(snapshot)
        replayed = algo.env_runner.collect(3)
        for key in expected.policy_batches:
            assert_nested_equal(dict(expected.policy_batches[key]), dict(replayed.policy_batches[key]))
        algo.set_state(snapshot)
        checkpoint = algo.save_to_path(str(tmp_path / "checkpoint"))
        changed = copy.deepcopy(snapshot)
        changed["learning_contract"]["gamma"] = 0.5
        with pytest.raises(ValueError, match="learning contract"):
            algo.set_state(changed)
        assert algo.outer_round == snapshot["stackelberg_runtime"]["outer_round"]
        second_round = algo.train()
        json.dumps(second_round["learners"], allow_nan=False)
        continued = algo.get_state()
        algo.restore_from_path(checkpoint)
        algo.train()
        assert_nested_equal(continued["learner_group"], algo.get_state()["learner_group"])
        assert algo.last_round == continued["stackelberg_runtime"]["last_round"]
        exported = algo.export_actors(tmp_path / "actors")
        from raymarl.algorithms.stackelberg.inference import ActorBundle

        bundle = ActorBundle(exported)
        assert all(not hasattr(module, "value") for module in bundle.modules.values())
        obs = algo.env_runner.env.reset(21)
        expected = algo.env_runner._draw("uc", obs, deterministic=True)["raw_latents"]
        np.testing.assert_allclose(bundle.uc_controls(obs), expected, rtol=0, atol=0)
    finally:
        algo.stop()


@pytest.mark.integration
@pytest.mark.parametrize("chosen", [0, 1, 2])
def test_common_response_state_and_selected_branch_continuation(ray_runtime, tmp_path, monkeypatch, chosen):
    from raymarl.algorithms.stackelberg import algorithm as implementation

    torch.set_num_threads(1)
    cfg = configuration(tmp_path)
    cfg.deviation_updates = 0
    cfg.uc_candidates = 2
    algo = cfg.build_algo()
    original_adapt = algo._adapt
    original_select = implementation.select_candidate
    responses = []

    def consumers(learner):
        return {
            "weights": {key: learner.module[key].get_state() for key in algo.consumer_ids},
            "adam": {
                name: opt.state_dict()
                for name, opt in learner._named_optimizers.items()
                if learner._optimizer_name_to_module[name] in algo.consumer_ids
            },
        }

    def adapt(module_ids, updates, **kwargs):
        assert_nested_equal(algo.learner_group.get_weights(), algo.env_runner.module.get_state())
        before = copy.deepcopy(
            {
                "sampling": algo.env_runner.get_state(components="sampling"),
                "consumers": [item.get() for item in algo.learner_group.foreach_learner(consumers)],
            }
        )
        original_adapt(module_ids, updates, **kwargs)
        if module_ids == algo.consumer_ids:
            responses.append((before, algo._snapshot()))

    def select(evaluated, **kwargs):
        _, gains = original_select(evaluated, **kwargs)
        return chosen, gains

    monkeypatch.setattr(algo, "_adapt", adapt)
    monkeypatch.setattr(implementation, "select_candidate", select)
    try:
        for round_index in range(2):
            responses.clear()
            algo.train()
            assert len(responses) == 4  # Main adaptation, incumbent and two proposal responses.
            left = responses[1][0]
            for response in responses[2:]:
                right = response[0]
                assert_nested_equal(left["consumers"], right["consumers"])
                a, b = left["sampling"]["sampling"], right["sampling"]["sampling"]
                for key in a:
                    if key != "environment":
                        assert_nested_equal(a[key], b[key])
                assert a["environment"].engine == b["environment"].engine
                assert a["environment"].profiles.fingerprint() == b["environment"].profiles.fingerprint()
            # Compare trajectories from each full common snapshot, including RNG.
            selected = algo._snapshot()
            algo.env_runner.set_state(left["sampling"])
            expected = algo.env_runner.collect(1)
            algo.env_runner.set_state(right["sampling"])
            actual = algo.env_runner.collect(1)
            for key in expected.policy_batches:
                assert_nested_equal(dict(expected[key]), dict(actual[key]))
            algo._restore(selected)
            committed = responses[chosen + 1][1]
            assert_nested_equal(committed["learner"], selected["learner"])
            assert_nested_equal(committed["rng"], selected["rng"])
            assert (
                algo.env_runner.env.engine.state
                == committed["runner"]["sampling"]["environment"].engine.state
            )
            main_steps = 8 * (round_index + 1)
            assert algo.last_round["main_periods_lifetime"] == main_steps
            assert algo.sampled_periods == 24 * (round_index + 1)
            assert algo.env_runner.env.engine.state.period_id == main_steps % 6
            assert algo.env_runner._scenario_seed == (2, 3)[(main_steps // 6) % 2]
    finally:
        algo.stop()


@pytest.mark.cuda
@pytest.mark.integration
def test_gpu_learner_cpu_runner_update(ray_runtime, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    algo = configuration(tmp_path, device="cuda").build_algo()
    try:
        algo._adapt(("uc",), 1, seed=2)
        assert next(algo.env_runner.module["uc"].parameters()).device.type == "cpu"
        devices = [
            result.get()
            for result in algo.learner_group.foreach_learner(lambda learner: str(learner._device))
        ]
        assert devices == ["cuda:0"]
    finally:
        algo.stop()
