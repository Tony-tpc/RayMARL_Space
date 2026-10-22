"""Recovery plumbing on tiny synthetic data; no Ausgrid training or test data."""

import csv
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.sn_energy_market.evaluation import cursor
from experiments.sn_energy_market.study_recovery import (
    StudyRecovery,
    combine_diagnostics,
)
from experiments.sn_energy_market.train import write_json
from RLAlgo.tests.conftest import ray_runtime  # noqa: F401
from RLAlgo.tests.test_stackelberg_algorithm import configuration


def test_diagnostics_count_failed_work_without_inventing_step_traces():
    def report(count):
        return {"coverage": {"refreshed_gae_checks": count}, "samples": [{"periods": count}],
                "updates": [count], "maximums": {"error": count},
                "gradients": {"uc/value": {"optimizer_steps": count, "raw_norm_min": count,
                                           "parameter_delta_max": count}},
                "reward_statistics": {"uc": {"count": count, "sum": count, "min": -count, "max": count}},
                "sampled_dates": {"round_8": [str(count)]}}
    combined = combine_diagnostics(report(1392), report(576))
    assert combined["gradients"]["uc/value"]["optimizer_steps"] == 1968
    assert combined["gradients"]["uc/value"]["raw_norm_min"] == 576
    assert combined["coverage"]["refreshed_gae_checks"] == 1968
    assert len(combined["samples"]) == len(combined["updates"]) == 2


def test_prebuild_verification_does_not_require_prepared_agent_order(tmp_path):
    checkpoint = tmp_path / "raymarl/trial/checkpoints/round7"
    checkpoint.mkdir(parents=True)
    write_json(checkpoint / "learning_contract.json", {})
    gradients = {agent + "/value": {"optimizer_steps": 1392}
                 for agent in ["uc", *(f"consumer_{i}" for i in range(16))]}
    for name, content in {
        "failure.json": {"diagnostics": {"gradients": gradients}},
        "source_manifest.json": {}, "status.json": {}, "experiment.json": {}, "protocol.json": {},
        "analysis/progress-check/failure-recovery-audit.json": {"complete_rounds": 7},
    }.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        write_json(tmp_path / name, content)
    (checkpoint.parent.parent / "batch_losses.jsonl").write_text("".join(
        json.dumps({"event": "batch", "batch_id": i, "samples": 384,
                    "branch_status": "retained", "phase": "consumer_adaptation"}) + "\n"
        for i in range(1, 30)))
    (tmp_path / "data").mkdir()
    with (tmp_path / "data/training-all-branches.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["batch_id"])
        writer.writerows([i] for i in range(1, 31) for _ in range(224 if i == 30 else 384))
    recovery = StudyRecovery(tmp_path, checkpoint)
    recovery.verify(SimpleNamespace(learning_contract=dict), 10)
    assert recovery.ledger["expected_actual_total"]["collected_training_periods"] == 15968
    assert (recovery.folder / "failed-attempt-training.csv").is_file()
    with pytest.raises(FileExistsError):
        recovery.verify(SimpleNamespace(learning_contract=dict), 10)


@pytest.mark.integration
def test_native_recovery_restores_adam_rng_and_only_appends_logical_prefix(ray_runtime, tmp_path):  # noqa: F811
    torch.set_num_threads(1)
    algorithm = configuration(tmp_path).build_algo()
    try:
        algorithm._adapt(algorithm.consumer_ids, 1, seed=2)
        algorithm.outer_round = algorithm._iteration = 7
        algorithm.completed_updates, algorithm.sampled_periods = 28, 10752
        checkpoint = algorithm.save_to_path(str(tmp_path / "saved"))
        write_json(tmp_path / "round-0007.json", {"end": cursor(algorithm.env_runner)})
        recovery = StudyRecovery(tmp_path, checkpoint)
        recovery.folder.mkdir(parents=True)
        recovery.ledger = {}
        recovery.old_log = recovery.folder / "original.jsonl"
        recovery.old_log.write_text("".join(json.dumps({"batch_id": i}) + "\n" for i in range(1, 30)))
        (tmp_path / "data").mkdir()
        with (recovery.folder / "failed-attempt-training.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["batch_id", "value"])
            writer.writeheader()
            writer.writerows({"batch_id": i, "value": i} for i in range(1, 31))
        with torch.no_grad():
            for parameter in algorithm.learner_group._learner.module["uc"].parameters():
                parameter.add_(1)
        recovery.restore(algorithm)
        assert algorithm.outer_round == algorithm.training_iteration == 7
        assert algorithm.sampled_periods == 10752 and algorithm.batch_loss_log.next_id == 29
        assert recovery.ledger["native_weights_and_adam_exact"] and recovery.ledger["checkpoint_exact_replay"]
        with (tmp_path / "data/training-all-branches.csv").open(encoding="utf-8-sig", newline="") as stream:
            assert [int(r["batch_id"]) for r in csv.DictReader(stream)] == list(range(1, 29))
        assert len(recovery.old_log.read_text().splitlines()) == 29
        assert len(algorithm.batch_loss_log.path.read_text().splitlines()) == 28
    finally:
        algorithm.stop()
