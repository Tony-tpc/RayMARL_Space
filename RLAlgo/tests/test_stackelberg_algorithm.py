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
            critic_epochs=2,
            minibatch_size=4,
            storage_path=str(path),
            checkpoint_interval=0,
        )
    )


def rows(values):
    return [{"seed": index, "returns": {"uc": value}, "score_complete": True} for index, value in enumerate(values)]


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


def test_candidate_configuration_requires_finite_complete_evaluation(tmp_path):
    cfg = configuration(tmp_path)
    cfg.evaluation_steps = 2
    with pytest.raises(ValueError, match="complete episodes"):
        cfg.validate()
    cfg.evaluation_steps = None
    env_cfg = cfg.scenario_factory.config
    cfg.scenario_factory = replace(cfg.scenario_factory, config=replace(
        env_cfg, system=replace(env_cfg.system, episode_mode="continuous")))
    with pytest.raises(ValueError, match="finite scenarios"):
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


@pytest.mark.parametrize("epochs", [0, -1, 1.5])
def test_critic_epoch_budget_requires_positive_integer(tmp_path, epochs):
    cfg = configuration(tmp_path).training(critic_epochs=epochs)
    with pytest.raises(ValueError, match="positive integers"):
        cfg.validate()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_critic_lambda_validation(tmp_path, value):
    with pytest.raises(ValueError, match="critic_lambda"):
        configuration(tmp_path).training(critic_lambda=value).validate()


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
    right.critic_epochs += 1
    assert left.learning_contract() != right.learning_contract()
    right.critic_epochs = left.critic_epochs
    right.critic_lambda = 0.8
    assert left.learning_contract() != right.learning_contract()
    right.critic_lambda = left.critic_lambda
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
        uc_after = algo.learner_group.get_weights()["uc"]
        assert_nested_equal(
            {key: value for key, value in uc_before.items() if key.startswith("actor.")},
            {key: value for key, value in uc_after.items() if key.startswith("actor.")},
        )
        assert any(
            not np.array_equal(value, uc_after[key])
            for key, value in uc_before.items() if key.startswith("value.")
        )
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
        changed = copy.deepcopy(snapshot)
        changed["learning_contract"]["update_protocol"] = "all_critics_then_selected_actors_common_consumer_state_v1"
        with pytest.raises(ValueError, match="learning contract"):
            algo.set_state(changed)
        assert algo.outer_round == snapshot["stackelberg_runtime"]["outer_round"]
        second_round = algo.train()
        json.dumps(second_round["learners"], allow_nan=False)
        continued = algo.get_state()
        algo.restore_from_path(checkpoint)
        algo.train()
        entries = [json.loads(line) for line in (Path(algo.logdir) / "batch_losses.jsonl").read_text().splitlines()]
        ids = [entry["batch_id"] for entry in entries if entry["event"] == "batch"]
        assert ids == list(range(1, len(ids) + 1))
        assert_nested_equal(continued["learner_group"], algo.get_state()["learner_group"])
        assert algo.last_round == continued["stackelberg_runtime"]["last_round"]
        exported = algo.export_actors(tmp_path / "actors")
        from raymarl.algorithms.stackelberg.inference import ActorBundle

        bundle = ActorBundle(exported)
        assert all(not hasattr(module, "value") for module in bundle.modules.values())
        obs = algo.env_runner.env.reset(21)
        bounds = algo.env_runner.env.uc_bounds()
        sample = algo.env_runner._draw("uc", obs, bounds, deterministic=True)
        expected = np.r_[sample["storage_mode"], sample["raw_latents"]]
        np.testing.assert_allclose(bundle.uc_controls(obs, bounds), expected, rtol=0, atol=0)
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
    cfg.inner_updates = cfg.response_updates = cfg.uc_updates_per_candidate = 2
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
        if module_ids == ("uc",):
            after = [item.get() for item in algo.learner_group.foreach_learner(consumers)]
            for old, new in zip(before["consumers"], after, strict=True):
                for key in algo.consumer_ids:
                    for name, value in old["weights"][key].items():
                        if name.startswith("actor."):
                            np.testing.assert_array_equal(value, new["weights"][key][name])
                    assert any(
                        not np.array_equal(value, new["weights"][key][name])
                        for name, value in old["weights"][key].items() if name.startswith("value.")
                    )
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
            main_steps = 16 * (round_index + 1)
            assert algo.last_round["main_periods_lifetime"] == main_steps
            assert algo.sampled_periods == 48 * (round_index + 1)
            assert algo.env_runner.env.engine.state.period_id == main_steps % 6
            assert algo.env_runner._scenario_seed == (2, 3)[(main_steps // 6) % 2]
            from experiments.sn_energy_market.plot_batch_losses import load_batches

            logged = load_batches(Path(algo.logdir) / "batch_losses.jsonl")
            assert [row["batch_id"] for row in logged] == list(range(1, 6 * (round_index + 1) + 1))
            for row in logged[-6:]:
                if row["phase"] == "uc_proposal":
                    assert row["retained_updates"]["uc"]["critic"] == (row["candidate_index"] <= chosen)
                    assert not row["sampling_retained"]
                    assert all(not row["retained_updates"][key]["critic"] for key in algo.consumer_ids)
                elif row["phase"] == "response":
                    assert row["sampling_retained"] == (row["candidate_index"] == chosen)
                    assert row["branch_status"] == ("retained" if row["candidate_index"] == chosen else "rolled_back")
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


@pytest.mark.integration
def test_phase_batch_records_targets_and_budget_do_not_roll_back(ray_runtime, tmp_path, monkeypatch):
    from experiments.sn_energy_market.plot_batch_losses import load_batches
    from experiments.sn_energy_market.training_audit import TrainingAudit

    torch.set_num_threads(1)
    cfg = configuration(tmp_path).training(
        inner_updates=2, uc_updates_per_candidate=2, response_updates=2,
        deviation_updates=0, critic_epochs=4, num_epochs=2,
    )
    # Tiny synthetic fixtures only; no historical energy training/validation/test data.
    cfg.test_seeds = ()
    algo = cfg.build_algo()
    audit = TrainingAudit(algo, tmp_path).install()
    path = Path(algo.logdir) / "batch_losses.jsonl"
    original_collect = algo.env_runner.collect
    original_diagnostics = algo._batch_diagnostics
    collected, before = [], []

    def collect(*args, **kwargs):
        if collected:
            assert len(load_batches(path)) == len(collected)
        batch = original_collect(*args, **kwargs)
        collected.append(copy.deepcopy(batch))
        return batch

    def diagnostics(batch, actor_ids=()):
        snapshot = algo._snapshot()
        result = original_diagnostics(batch, actor_ids)
        after = algo._snapshot()
        assert_nested_equal(snapshot["learner"], after["learner"])
        assert_nested_equal(snapshot["rng"], after["rng"])
        assert snapshot["runner"]["sampling"]["environment"].engine == after["runner"]["sampling"]["environment"].engine
        if not actor_ids:
            before.append(result)
        return result

    monkeypatch.setattr(algo.env_runner, "collect", collect)
    monkeypatch.setattr(algo, "_batch_diagnostics", diagnostics)
    try:
        for index in range(3):
            audit.phase = f"round_{index + 1}"
            algo.train()
        assert len(audit.updates) == 24
        assert audit.coverage["frozen_optimizer_checks"] > 0
        records = load_batches(path)
        assert [item["batch_id"] for item in records] == list(range(1, 13))
        assert algo.completed_updates == 12 and algo.sampled_periods == 96
        assert len(collected) == len(before) == 12
        assert all(batch.count == 8 for batch in collected)
        for key in algo.config.agent_order:
            critic_steps = [step for step in audit.optimizer_steps if step["agent"] == key and step["role"] == "value"]
            assert len(critic_steps) == 3 * 4 * 4 * (8 // 4)
        uc_steps = [step for step in audit.optimizer_steps if step["agent"] == "uc" and step["role"] == "actor"]
        assert len(uc_steps) == 3 * 2 * (8 // 4)
        assert audit.coverage["refreshed_gae_checks"] > 0
        for item, measured in zip(records, before, strict=True):
            assert item["critic_epochs"] == 4 and item["actor_epochs"] == 2
            assert item["branch_status"] != "pending"
            for key, values in item["modules"].items():
                mode_diagnostics = values["mode_diagnostics"]
                if key == "uc" and key in item["actor_ids"]:
                    assert mode_diagnostics["unique_samples"] == 8
                    assert sum(row["count"] for row in mode_diagnostics["soc_advantages"]) == 8
                    assert mode_diagnostics["summary"]["sample_visits"] == 16
                    assert len(mode_diagnostics["minibatches"]) == 4
                    assert all(row["adam_step_after"] == row["adam_step_before"] + 1
                               for row in mode_diagnostics["minibatches"])
                else:
                    assert mode_diagnostics is None
                assert values["critic_loss_before"] == measured[key]["critic_loss"]
                for suffix in ("before", "after"):
                    assert values[f"critic_loss_{suffix}"] == pytest.approx(
                        0.5 * (values[f"critic_rmse_{suffix}"] / values["value_scale"]) ** 2
                    )
                    assert (values[f"policy_loss_{suffix}"] is None) == (key not in item["actor_ids"])
        for start in (0, 4, 8):
            assert [item["phase"] for item in records[start:start + 4]] == (
                ["consumer_adaptation", "uc_proposal", "response", "response"]
            )
        # Candidate snapshot restore never reuses audit IDs.
        saved = algo._snapshot()
        algo._restore(saved)
        algo._adapt(algo.consumer_ids, 1)
        assert load_batches(path)[-1]["batch_id"] == 13
    finally:
        audit.close()
        algo.stop()
