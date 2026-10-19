"""Training and validation can run without opening held-out test profiles."""

import json
import os
import shutil
import sys
from types import SimpleNamespace

import pytest

from experiments.sn_energy_market import train
from experiments.sn_energy_market.tests.test_profile_dataset import make_dataset


@pytest.fixture(scope="module")
def selected_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("reserved-test")
    manifest = make_dataset(root / "complete")
    selected = root / "selected"
    selected.mkdir()
    for name in ("manifest.json", "train.npz", "validation.npz"):
        shutil.copy2(manifest.parent / name, selected / name)
    return selected / "manifest.json"


def test_skip_test_keeps_only_train_and_validation_scenarios(selected_dataset):
    args = train.arguments(["--dataset", str(selected_dataset), "--skip-test"])
    config, budget, splits = train.experiment_settings(args)
    assert set(splits) == {"train", "validation"}
    assert len(splits["train"]) == 12
    assert len(splits["validation"]) == 12
    factory = train.scenario_factory(args, config, splits)
    assert factory.dataset.splits == ("train", "validation")
    report = train.provenance(config, budget, splits, factory, args, {})
    assert set(report["profiles"]) == {"train", "validation"}
    assert report["test_protocol"] == "reserved; no test profiles loaded or evaluated"
    assert report["scenario_contract"]["loaded_splits"] == ["train", "validation"]


def test_skip_test_data_validation_does_not_load_test_or_ray(
    selected_dataset, tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "ray", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    report = train.main(
        [
            "--dataset",
            str(selected_dataset),
            "--skip-test",
            "--validate-data",
            "--output",
            str(tmp_path),
        ]
    )
    assert report["training_started"] is False
    assert report["test_reserved"] is True
    assert set(report["splits"]) == {"train", "validation"}
    saved = json.loads((tmp_path / "data_contract.json").read_text(encoding="utf-8"))
    assert saved["test_reserved"] is True
    assert set(saved["splits"]) == {"train", "validation"}


def test_skip_test_also_reserves_synthetic_test_scenarios():
    args = train.arguments(["--smoke", "--skip-test"])
    config, _, splits = train.experiment_settings(args)
    assert set(splits) == {"train", "validation"}
    assert train.scenario_factory(args, config, splits).seed_splits == splits


@pytest.mark.parametrize("skip_test", [False, True])
def test_training_entry_routes_final_evaluation_explicitly(
    tmp_path, monkeypatch, skip_test
):
    configured, evaluated = {}, []
    algorithm = SimpleNamespace(
        iteration=0,
        restore_from_path=lambda path: None,
        evaluate_pair=lambda seeds: evaluated.append(tuple(seeds)) or {"rows": []},
        save_to_path=lambda path: path,
        export_actors=lambda path: None,
        stop=lambda: None,
    )

    class Config:
        def scenarios(self, factory, **splits):
            configured.update(splits)
            return self

        def debugging(self, **kwargs):
            return self

        rl_module = learners = training = debugging

        def build_algo(self):
            return algorithm

        def learning_contract(self):
            return {}

    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "provenance", lambda *args: {})
    for name in ("TMP", "TEMP", "RAY_USAGE_STATS_ENABLED", "RAY_AUTH_TOKEN_PATH"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(init=lambda **kwargs: None, shutdown=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "raymarl.algorithms.stackelberg",
        SimpleNamespace(StackelbergConfig=Config),
    )
    output = tmp_path / "report"
    argv = [
        "--smoke",
        "--evaluate-only",
        "--resume",
        str(tmp_path / "checkpoint"),
        "--output",
        str(output),
    ]
    train.main(argv + (["--skip-test"] if skip_test else []))
    split = "validation" if skip_test else "test"
    filename = "validation_evaluation.json" if skip_test else "held_out_evaluation.json"
    assert evaluated == [configured[f"{split}_seeds"]]
    assert configured["test_seeds"] == (() if skip_test else (20042, 20043))
    assert (output / filename).exists()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["evaluation_split"] == split
    assert result["test_reserved"] is skip_test
    if skip_test:
        assert not (output / "held_out_evaluation.json").exists()
