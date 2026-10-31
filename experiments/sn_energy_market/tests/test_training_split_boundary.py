"""The production recipe can be built with the held-out test file absent."""

import shutil

import pytest

from experiments.sn_energy_market import runtime
from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)
from experiments.sn_energy_market.settings import inventory_valued_config
from experiments.sn_energy_market.tests.test_profile_dataset import make_dataset


@pytest.fixture(scope="module")
def selected_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("reserved-test")
    manifest = make_dataset(root / "complete")
    selected = root / "datasets/ausgrid/processed"
    selected.mkdir(parents=True)
    for name in ("manifest.json", "train.npz", "validation.npz"):
        shutil.copy2(manifest.parent / name, selected / name)
    return root


def test_production_configuration_does_not_open_test_profiles(selected_root, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", selected_root)
    config = runtime.configuration(tmp_path)
    factory = config.scenario_factory
    assert factory.dataset.splits == ("train", "validation")
    assert set(factory.seed_splits) == {"train", "validation"}
    assert len(config.train_seeds) == 12 and config.test_seeds == ()
    assert config.learning_contract()["scenario"]["loaded_splits"] == ["train", "validation"]
    assert not (selected_root / "datasets/ausgrid/processed/test.npz").exists()


def test_reserved_factory_has_no_test_windows(selected_root):
    dataset = ProfileDataset(selected_root / "datasets/ausgrid/processed/manifest.json",
                             splits=("train", "validation"))
    factory = DatasetEnvFactory(dataset, inventory_valued_config(dataset.environment_config()), seed=42)
    with pytest.raises(KeyError, match="20042"):
        factory(20042)
