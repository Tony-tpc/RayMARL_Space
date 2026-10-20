import hashlib
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest
from snmarket import EnvironmentConfig
from snmarket.action import map_consumer_controls

from experiments.sn_energy_market import train
from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)


def make_dataset(directory, *, holdout_multiplier=1.0):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "test_measured_profiles",
        "household_ids": [23, 5],
        "resolution_minutes": 30,
        "time_policy": "fixed_model_clock",
        "units": {"loads_kw": "kW", "der_kw": "kW"},
        "splits": {},
    }
    for index, split in enumerate(("train", "validation", "test")):
        start = datetime.fromisoformat(f"{2010 + index}-07-01T00:00:00+10:00")
        end = start.replace(year=start.year + 1)
        periods = int((end - start).total_seconds() / 1800)
        timestamps = np.array(
            [
                (start + timedelta(minutes=30 * step)).isoformat()
                for step in range(periods)
            ]
        )
        loads = np.tile([1.0, 2.0], (periods, 1)) * (holdout_multiplier if index else 1)
        der = np.ones(periods)
        path = directory / f"{split}.npz"
        np.savez_compressed(path, loads_kw=loads, der_kw=der, timestamps=timestamps)
        manifest["splits"][split] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "start_datetime": start.isoformat(),
            "end_datetime": end.isoformat(),
            "periods": periods,
        }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("profiles"))


def test_all_months_and_fixed_household_order_with_leap_year(dataset_path):
    data = ProfileDataset(dataset_path)
    factory = DatasetEnvFactory(data, seed=80)
    assert data.household_ids == (23, 5)
    assert {key: len(value) for key, value in factory.seed_splits.items()} == {
        "train": 12,
        "validation": 12,
        "test": 12,
    }
    assert data.windows("train")[0].name == "2010-07"
    assert (
        next(
            window for window in data.windows("validation") if window.name == "2012-02"
        ).periods
        == 29 * 48
    )
    for split, seeds in factory.seed_splits.items():
        for seed, window in zip(seeds, data.windows(split)):
            env = factory(seed)
            assert env.config.system.horizon_steps == window.periods
            assert env.engine.profiles.metadata["household_ids"] == [23, 5]
            assert env.engine.profiles.metadata["split"] == split
            np.testing.assert_array_equal(env.engine.profiles.loads_kw[0], [1, 2])
    all_seeds = [seed for seeds in factory.seed_splits.values() for seed in seeds]
    assert len(set(all_seeds)) == 36
    with pytest.raises(KeyError):
        factory(12345)


def test_training_and_validation_work_without_loading_test_data(dataset_path, tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    for name in ("manifest.json", "train.npz", "validation.npz"):
        shutil.copy2(dataset_path.parent / name, selected / name)
    dataset = ProfileDataset(selected / "manifest.json", splits=("validation", "train"))
    factory = DatasetEnvFactory(dataset, seed=80)
    assert dataset.splits == ("train", "validation")
    assert set(dataset.manifest["splits"]) == {"train", "validation", "test"}
    assert factory.seed_splits == {
        "train": tuple(range(80, 92)),
        "validation": tuple(range(10080, 10092)),
    }
    for split, seeds in factory.seed_splits.items():
        env = factory(seeds[0])
        env.reset(seeds[0])
        assert env.engine.profiles.metadata["split"] == split
    contract = factory.get_contract()
    assert contract["loaded_splits"] == ["train", "validation"]
    assert (
        contract["manifest_sha256"]
        == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    )
    with pytest.raises(KeyError):
        dataset.windows("test")
    with pytest.raises(KeyError):
        factory(20080)
    with pytest.raises(FileNotFoundError):
        ProfileDataset(selected / "manifest.json")


def test_thirty_minute_conversion_preserves_physical_durations_and_coefficients(
    dataset_path,
):
    base = EnvironmentConfig()
    base = replace(base, rewards=replace(base.rewards, external_cost_per_period=0.3))
    dataset = ProfileDataset(dataset_path)
    config = dataset.environment_config(base)
    assert config.system.dt_hours == 0.5
    assert config.market.shift_deadline_steps == 48
    assert config.market.dr_event_intervals == ((34, 40),)
    assert config.rewards.uc_due_rate_per_kwh == 2.0
    assert config.rewards.external_cost_per_period == 0.3
    assert (
        config.rewards.consumer_wait_rate_per_kwh_hour
        == base.rewards.consumer_wait_rate_per_kwh_hour
    )
    assert (
        config.rewards.actual_compensation_per_kwh
        == base.rewards.actual_compensation_per_kwh
    )
    assert config.system.grid_import_max_kw == base.system.grid_import_max_kw
    assert config.system.ess_energy_capacity_kwh == base.system.ess_energy_capacity_kwh
    assert config.market.shift_backlog_limit_kwh == base.market.shift_backlog_limit_kwh
    calibrated = dataset.environment_config()
    assert calibrated.system.grid_import_max_kw == 3.0
    assert calibrated.system.ess_energy_capacity_kwh == 1.5
    np.testing.assert_allclose(calibrated.market.shift_backlog_limit_kwh, [4.8, 9.6])


def test_calibration_ignores_validation_and_test_values(dataset_path, tmp_path):
    ordinary = ProfileDataset(dataset_path)
    extreme = ProfileDataset(
        make_dataset(tmp_path / "extreme", holdout_multiplier=1000)
    )
    assert ordinary.calibration() == extreme.calibration()
    assert ordinary.environment_config() == extreme.environment_config()
    assert (
        DatasetEnvFactory(ordinary).get_contract()
        != DatasetEnvFactory(extreme).get_contract()
    )


def test_contract_is_portable_and_binds_dates_households_and_seed_assignment(
    dataset_path, tmp_path
):
    copied = tmp_path / "copied"
    shutil.copytree(dataset_path.parent, copied)
    original = DatasetEnvFactory(ProfileDataset(dataset_path))
    relocated = DatasetEnvFactory(ProfileDataset(copied / "manifest.json"))
    assert original.get_contract() == relocated.get_contract()
    assert (
        original.get_contract()
        != DatasetEnvFactory(original.dataset, seed=1).get_contract()
    )
    contract = original.get_contract()
    assert contract["household_ids"] == [23, 5]
    assert (
        contract["seed_windows"]["42"]["start_datetime"] == "2010-07-01T00:00:00+10:00"
    )
    assert "profile_dataset.py" in contract["source_sha256"]
    contract["household_ids"].append(99)
    assert original.get_contract()["household_ids"] == [23, 5]


def test_corrupted_data_rejected_before_environment_creation(dataset_path, tmp_path):
    copied = tmp_path / "corrupted"
    shutil.copytree(dataset_path.parent, copied)
    with (copied / "train.npz").open("ab") as stream:
        stream.write(b"unexpected change")
    with pytest.raises(ValueError, match="checksum"):
        ProfileDataset(copied / "manifest.json")


def test_energy_inputs_are_not_silently_interpreted_as_power(dataset_path, tmp_path):
    copied = tmp_path / "wrong_units"
    shutil.copytree(dataset_path.parent, copied)
    path = copied / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["units"]["loads_kw"] = "kWh"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="power in kW"):
        ProfileDataset(path)


def test_chronological_overlap_rejected_even_with_valid_file_hashes(
    dataset_path, tmp_path
):
    copied = tmp_path / "overlap"
    shutil.copytree(dataset_path.parent, copied)
    path = copied / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["validation"] = manifest["splits"]["train"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered and disjoint"):
        ProfileDataset(path)


def test_monthly_episode_preserves_state_at_midnight_and_snapshot(dataset_path):
    factory = DatasetEnvFactory(ProfileDataset(dataset_path))
    env = factory(42)
    env.reset(42)
    for _ in range(49):
        env.publish([1, 0, 0, 0, 0, 0])
        actions = {
            key: map_consumer_controls([0, 1, 0.5, 0], bound)
            for key, bound in env.engine.local_bounds().items()
        }
        result = env.step(actions)
    assert not result.terminated
    assert env.engine.state.period_id == 49
    assert env.engine.state.timestamp.day == 2
    assert any(
        consumer.monthly_uc_energy_kwh > 0 for consumer in env.engine.state.consumers
    )
    snapshot = env.get_state()
    env.publish([0.8, 0, 0, 0, 0, 0])
    actions = {
        key: map_consumer_controls([0.1, 0.5, 0.5, 0.5], bound)
        for key, bound in env.engine.local_bounds().items()
    }
    expected = env.step(actions)
    env.set_state(snapshot)
    env.publish([0.8, 0, 0, 0, 0, 0])
    actual = env.step(actions)
    assert actual.rewards == expected.rewards
    assert actual.metrics == expected.metrics


def test_training_entry_uses_all_months_and_converted_learning_parameters(dataset_path):
    args = train.arguments(["--dataset", str(dataset_path)])
    config, budget, splits = train.experiment_settings(args)
    assert len(splits["train"]) == 12
    assert len(splits["validation"]) == len(splits["test"]) == 12
    assert config.system.num_consumers == 2
    assert budget["gamma"] == pytest.approx(0.9801)
    assert budget["lambda_gae"] == pytest.approx(0.9025)
    assert budget["critic_lambda"] == 1.0
    assert budget["rollout_steps"] == 64
    factory = train.scenario_factory(args, config, splits)
    assert factory.seed_splits == splits
    with pytest.raises(ValueError, match="complete calendar months"):
        train.experiment_settings(
            train.arguments(["--dataset", str(dataset_path), "--horizon", "48"])
        )
    with pytest.raises(ValueError, match="fixed household"):
        train.experiment_settings(
            train.arguments(["--dataset", str(dataset_path), "--consumers", "3"])
        )


def test_data_validation_never_imports_or_starts_ray(
    dataset_path, tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "ray", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    output = tmp_path / "report"
    report = train.main(
        ["--dataset", str(dataset_path), "--validate-data", "--output", str(output)]
    )
    assert report["training_started"] is False
    assert report["status"] == "data_contract_validated"
    assert (output / "data_contract.json").exists()
    assert len(report["splits"]["test"]) == 12
