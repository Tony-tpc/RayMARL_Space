"""Batch loss history survives candidate/checkpoint rollback without fabricating points."""

import json

import pytest

from raymarl.algorithms.stackelberg.batch_log import BatchLossLog


def row(phase, candidate=None):
    return {
        "phase": phase, "candidate_index": candidate,
        "actor_ids": ["uc"] if phase == "uc_proposal" else ["consumer"],
        "modules": {"uc": {}, "consumer": {}},
    }


def test_batch_is_flushed_before_selection_and_resolution_only_appends(tmp_path):
    from experiments.sn_energy_market.plot_batch_losses import load_batches

    path = tmp_path / "batch_losses.jsonl"
    log = BatchLossLog(path)
    log.record(row("consumer_adaptation"))
    log.record(row("uc_proposal", 1))
    log.record(row("response", 0))
    log.record(row("response", 1))
    original = path.read_bytes()
    pending = load_batches(path)
    assert [item["batch_id"] for item in pending] == [1, 2, 3, 4]
    assert [item["branch_status"] for item in pending] == ["pending"] * 4
    log.resolve(0)
    assert path.read_bytes().startswith(original)
    resolved = load_batches(path)
    assert [item["branch_status"] for item in resolved] == ["retained", "rolled_back", "retained", "rolled_back"]
    reopened = BatchLossLog(path)
    reopened.record(row("deviation"))
    last = load_batches(path)[-1]
    assert last["batch_id"] == 5 and last["branch_status"] == "rolled_back"
    assert not any(any(roles.values()) for roles in last["retained_updates"].values())


def test_plot_reader_does_not_invent_missing_batches(tmp_path):
    from experiments.sn_energy_market.plot_batch_losses import load_batches

    path = tmp_path / "batch_losses.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No measured batch"):
        load_batches(path)
    entry = {"event": "batch", "batch_id": 2, **row("adapt")}
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert [item["batch_id"] for item in load_batches(path)] == [2]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")
    with pytest.raises(ValueError, match="Duplicate batch"):
        load_batches(path)


def test_whole_round_rejection_marks_adaptation_and_candidate_zero_discarded(tmp_path):
    from experiments.sn_energy_market.plot_batch_losses import load_batches

    log = BatchLossLog(tmp_path / "batch_losses.jsonl")
    for phase, candidate in (("consumer_adaptation", None), ("uc_proposal", 1), ("response", 0)):
        log.record(row(phase, candidate))
    log.resolve(None)
    for record in load_batches(log.path):
        assert record["branch_status"] == "rolled_back" and not record["sampling_retained"]
        assert not any(any(roles.values()) for roles in record["retained_updates"].values())


def test_interrupted_round_remains_pending_when_a_later_round_resolves(tmp_path):
    from experiments.sn_energy_market.plot_batch_losses import load_batches

    path = tmp_path / "batch_losses.jsonl"
    log = BatchLossLog(path)
    log.record(row("response", 0))
    log.begin_round()
    log.record(row("response", 0))
    log.resolve(0)
    assert [item["branch_status"] for item in load_batches(path)] == ["pending", "retained"]
