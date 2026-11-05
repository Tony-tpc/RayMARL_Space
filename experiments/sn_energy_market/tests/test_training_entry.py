"""The production entry rejects invalid budgets and existing runs."""

import os

import pytest

from experiments.sn_energy_market import run


@pytest.mark.parametrize("rounds", [0, -1])
def test_training_requires_positive_round_budget(tmp_path, rounds):
    with pytest.raises(ValueError, match="rounds must be positive"):
        run.main(tmp_path, rounds)
    assert not list(tmp_path.iterdir())


def test_existing_run_preserves_evidence_without_starting_ray(tmp_path, monkeypatch):
    from experiments.sn_energy_market import runtime

    output = tmp_path / "training"
    output.mkdir()
    evidence = {"status.json": '{"status":"completed"}', "result.json": '{"rounds":72}'}
    for name, content in evidence.items():
        (output / name).write_text(content, encoding="utf-8")
    for name in ("TMP", "TEMP", "PYTHONDONTWRITEBYTECODE", "RAY_AUTH_MODE", "RAY_AUTH_TOKEN_PATH",
                 "RAY_USAGE_STATS_ENABLED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, ""))

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing run must be rejected before starting Ray")

    monkeypatch.setattr(runtime, "ray_session", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        run.main(tmp_path, 72)
    for name, content in evidence.items():
        assert (output / name).read_text(encoding="utf-8") == content


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_device_controls_ray_resources(tmp_path, monkeypatch, device):
    import ray
    import torch

    from experiments.sn_energy_market.runtime import ray_session

    for name in ("TMP", "TEMP", "CUBLAS_WORKSPACE_CONFIG", "RAY_USAGE_STATS_ENABLED",
                 "RAY_AUTH_MODE", "RAY_AUTH_TOKEN_PATH"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")
    monkeypatch.setattr(ray, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(ray, "shutdown", lambda: calls.append("shutdown"))
    with ray_session(tmp_path, device=device):
        assert calls[0]["num_gpus"] == int(device == "cuda")
    assert calls[-1] == "shutdown"
