import os
import sys
from types import SimpleNamespace

import pytest

from experiments.sn_energy_market import train
from experiments.sn_energy_market.train import arguments, experiment_settings
from experiments.sn_energy_market.training_env import MarketEnvFactory


def test_smoke_contains_shift_and_event_decisions_and_separate_seed_sets():
    config, budget, splits = experiment_settings(arguments(["--smoke"]))
    assert config.system.num_consumers == 2
    assert budget["critic_lambda"] == 1.0
    assert config.market.shift_deadline_steps < config.system.horizon_steps
    assert config.system.start_datetime == "2026-01-01T17:00:00+08:00"
    assert (
        budget["inner_updates"]
        == budget["response_updates"]
        == budget["candidates"]
        == 1
    )
    assert all(
        not set(left).intersection(right)
        for key, left in splits.items()
        for other, right in splits.items()
        if key != other
    )
    factory = MarketEnvFactory(config, seed_splits=splits)
    for name, seeds in splits.items():
        assert factory(seeds[0]).engine.profiles.metadata["split"] == name


def test_default_training_keeps_document_deadline_and_allows_new_tasks():
    config, _, _ = experiment_settings(arguments([]))
    assert config.system.horizon_steps == 288
    assert config.market.shift_deadline_steps == 96


def test_explicit_budget_and_dimensions_override_smoke_defaults():
    config, budget, _ = experiment_settings(
        arguments(
            [
                "--smoke",
                "--consumers",
                "3",
                "--horizon",
                "24",
                "--deadline",
                "4",
                "--outer-rounds",
                "3",
                "--rollout-steps",
                "12",
                "--response-updates",
                "2",
                "--critic-epochs",
                "5",
                "--critic-lambda",
                "0.8",
            ]
        )
    )
    assert config.system.num_consumers == 3
    assert config.system.horizon_steps == 24
    assert config.market.shift_deadline_steps == 4
    assert budget["outer_rounds"] == 3
    assert budget["rollout_steps"] == 12
    assert budget["response_updates"] == 2
    assert budget["critic_epochs"] == 5
    assert budget["critic_lambda"] == 0.8


@pytest.mark.parametrize("rounds", ["0", "-1"])
def test_training_requires_positive_outer_round_budget(rounds):
    with pytest.raises(ValueError, match="outer-rounds must be positive"):
        experiment_settings(arguments(["--smoke", "--outer-rounds", rounds]))


def test_frozen_evaluation_accepts_zero_training_rounds():
    _, budget, _ = experiment_settings(
        arguments(
            [
                "--smoke",
                "--evaluate-only",
                "--resume",
                "checkpoint",
                "--outer-rounds",
                "0",
            ]
        )
    )
    assert budget["outer_rounds"] == 0


def test_rejected_resume_preserves_existing_evidence_without_starting_ray(
    tmp_path, monkeypatch
):
    output = tmp_path / "existing"
    output.mkdir()
    evidence = {
        "experiment.json": '{"seed": 42}',
        "result.json": '{"status": "completed"}',
    }
    for name, content in evidence.items():
        (output / name).write_text(content, encoding="utf-8")
    calls = []

    class Config:
        def scenarios(self, *args, **kwargs):
            return self

        debugging = rl_module = learners = training = scenarios

        def build_algo(self):
            return self

        def restore_from_path(self, path):
            calls.append("restore")
            raise ValueError("Checkpoint learning contract differs")

        def stop(self):
            calls.append("stop")

    monkeypatch.setattr(train, "ROOT", tmp_path)
    for name in ("TMP", "TEMP", "RAY_USAGE_STATS_ENABLED", "RAY_AUTH_TOKEN_PATH"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            init=lambda **kwargs: calls.append("init"),
            shutdown=lambda: calls.append("shutdown"),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(set_num_threads=lambda value: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "raymarl.algorithms.stackelberg",
        SimpleNamespace(StackelbergConfig=Config),
    )
    with pytest.raises(ValueError, match="learning contract differs"):
        train.main(
            ["--smoke", "--resume", str(output / "checkpoint"), "--output", str(output)]
        )
    assert calls == ["init", "restore", "stop", "shutdown"]
    for name, content in evidence.items():
        assert (output / name).read_text(encoding="utf-8") == content
