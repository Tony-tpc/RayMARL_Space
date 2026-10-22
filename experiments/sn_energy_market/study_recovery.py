"""Resume a saved study boundary; keep failed work separate from the comparison path."""

import csv
import hashlib
import json
import pickle
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.sn_energy_market.evaluation import ROOT, cursor
from experiments.sn_energy_market.plot_batch_losses import load_batches
from experiments.sn_energy_market.train import write_json
from experiments.sn_energy_market.training_audit import assert_same_state


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def combine_diagnostics(before, after):
    """Actual executed work, including the failed branch; never fabricate step traces."""
    result = {"coverage": dict(Counter(before["coverage"]) + Counter(after["coverage"])),
              "samples": before["samples"] + after["samples"],
              "updates": before["updates"] + after["updates"],
              "maximums": {}, "gradients": {}, "reward_statistics": {}, "sampled_dates": {}}
    for key in before["maximums"].keys() | after["maximums"].keys():
        result["maximums"][key] = max(before["maximums"].get(key, 0), after["maximums"].get(key, 0))
    for key in before["gradients"].keys() | after["gradients"].keys():
        left, right = before["gradients"].get(key, {}), after["gradients"].get(key, {})
        result["gradients"][key] = {}
        for metric in left.keys() | right.keys():
            values = [row[metric] for row in (left, right) if metric in row]
            operation = min if metric.endswith("_min") else max if metric.endswith("_max") else sum
            result["gradients"][key][metric] = operation(values)
    for key in before["reward_statistics"].keys() | after["reward_statistics"].keys():
        rows = [part["reward_statistics"][key] for part in (before, after) if key in part["reward_statistics"]]
        result["reward_statistics"][key] = {
            "count": sum(r["count"] for r in rows), "sum": sum(r["sum"] for r in rows),
            "min": min(r["min"] for r in rows), "max": max(r["max"] for r in rows),
        }
    for key in before["sampled_dates"].keys() | after["sampled_dates"].keys():
        result["sampled_dates"][key] = sorted(set(before["sampled_dates"].get(key, []))
                                              | set(after["sampled_dates"].get(key, [])))
    return result


class StudyRecovery:
    """One authorized recovery from the completed round-7 checkpoint to round 10."""

    def __init__(self, output, checkpoint):
        self.output, self.checkpoint = Path(output), Path(checkpoint).resolve()
        self.folder = self.output / "recovery/from-round-0007"
        self.round, self.batches = 7, 28

    def verify(self, cfg, outer_rounds):
        assert outer_rounds == 10
        assert self.checkpoint.is_relative_to(self.output.resolve())
        assert not (self.output / "report.json").exists()
        assert not (self.output / "round-0008.json").exists()
        assert read_json(self.checkpoint / "learning_contract.json") == cfg.learning_contract()
        manifest = read_json(self.output / "source_manifest.json")
        # All loaded algorithm/environment sources must match the interrupted run.
        for relative, digest in manifest.items():
            if relative.startswith(("RLAlgo/", "environments/")):
                assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
        self.previous = read_json(self.output / "failure.json")["diagnostics"]
        self.failure_audit = read_json(self.output / "analysis/progress-check/failure-recovery-audit.json")
        assert self.failure_audit["complete_rounds"] == self.round
        self.old_log = self.checkpoint.parent.parent / "batch_losses.jsonl"
        rows = load_batches(self.old_log)
        assert [r["batch_id"] for r in rows] == list(range(1, 30))
        assert all(r["samples"] == 384 for r in rows)
        assert all(r["branch_status"] != "pending" for r in rows[:self.batches])
        assert rows[-1]["phase"] == "consumer_adaptation"
        value_gradients = {key: value for key, value in self.previous["gradients"].items()
                           if key.endswith("/value")}
        assert len(value_gradients) == 17 and "uc/value" in value_gradients
        assert all(row["optimizer_steps"] == 1392 for row in value_gradients.values())
        with (self.output / "data/training-all-branches.csv").open(encoding="utf-8-sig", newline="") as stream:
            counts = Counter(int(r["batch_id"]) for r in csv.DictReader(stream))
        assert counts == Counter({**{i: 384 for i in range(1, 30)}, 30: 224})
        self.folder.mkdir(parents=True, exist_ok=False)  # Refuse a duplicate recovery launch.
        for name in ("status.json", "failure.json", "experiment.json", "protocol.json", "source_manifest.json"):
            shutil.copy2(self.output / name, self.folder / ("original-" + name))
        shutil.copy2(self.output / "data/training-all-branches.csv", self.folder / "failed-attempt-training.csv")
        shutil.copy2(self.old_log, self.folder / "failed-attempt-batch-losses.jsonl")
        self.ledger = {
            "status": "prepared", "checkpoint": str(self.checkpoint), "resume_after_round": self.round,
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "target_round": outer_rounds, "user_authorized_recomputation": True,
            "algorithm_and_environment_sources_match_original": True,
            "checkpoint_sha256": {str(p.relative_to(self.checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in self.checkpoint.rglob("*") if p.is_file()},
            "failed_attempt": {"completed_learning_batches": 29, "learning_periods": 11136,
                               "collected_training_periods": 11360, "partial_batch_periods": 224,
                               "critic_steps_per_module": 1392},
            "discarded_since_checkpoint": {"learning_batches": 1, "learning_periods": 384,
                                           "collected_training_periods": 608, "critic_steps_per_module": 48,
                                           "consumer_actor_steps_per_module": 24, "uc_actor_steps": 0},
            "remaining": {"rounds": 3, "learning_batches": 12, "learning_periods": 4608,
                          "main_periods": 2304, "critic_steps_per_module": 576},
            "effective_budget": {"rounds": 10, "learning_batches": 40, "learning_periods": 15360,
                                 "main_periods": 7680, "critic_steps_per_module": 1920},
            "expected_actual_total": {"learning_batches": 41, "learning_periods": 15744,
                                      "collected_training_periods": 15968, "critic_steps_per_module": 1968,
                                      "consumer_actor_steps_per_module": 744, "uc_actor_steps": 240},
            "optimizer_step_trace_scope": "optimizer_steps.json records resumed rounds 8-10 only; original summaries preserved",
        }
        write_json(self.folder / "ledger.json", self.ledger)

    def restore(self, algorithm):
        started = time.monotonic()
        algorithm.restore_from_path(str(self.checkpoint))
        assert algorithm.outer_round == algorithm.training_iteration == self.round
        assert algorithm.completed_updates == self.batches and algorithm.sampled_periods == 10752
        assert cursor(algorithm.env_runner) == read_json(self.output / "round-0007.json")["end"]
        learner_state = pickle.loads((self.checkpoint / "learner_group/learner/state.pkl").read_bytes())
        learner = algorithm.learner_group._learner
        assert_same_state(learner_state["optimizer"], learner.get_state()["optimizer"])
        for agent in algorithm.config.agent_order:
            weights = pickle.loads((self.checkpoint / f"learner_group/learner/rl_module/{agent}/module_state.pkl").read_bytes())
            assert_same_state(weights, learner.module[agent].get_state())
        saved_state = pickle.loads((self.checkpoint / "algorithm_state.pkl").read_bytes())
        assert_same_state(saved_state["raymarl_rng"], algorithm.get_state()["raymarl_rng"])
        expected = algorithm.env_runner.collect(8)
        algorithm.restore_from_path(str(self.checkpoint))
        actual = algorithm.env_runner.collect(8)
        for key, data in expected.policy_batches.items():
            for name, values in data.items():
                np.testing.assert_array_equal(values, actual[key][name])
        algorithm.restore_from_path(str(self.checkpoint))
        assert_same_state(saved_state["raymarl_rng"], algorithm.get_state()["raymarl_rng"])
        self.ledger.update(status="restored", native_weights_and_adam_exact=True, checkpoint_exact_replay=True,
                           replay_diagnostic_periods=16, replay_diagnostics_rewound=True,
                           restore_and_replay_seconds=time.monotonic() - started,
                           restored_at=datetime.now(timezone.utc).isoformat(),
                           new_algorithm_logdir=str(algorithm.logdir))
        # New log contains the logical path. Original failed-attempt log stays untouched.
        prefix = [line for line in self.old_log.read_text(encoding="utf-8").splitlines()
                  if json.loads(line)["batch_id"] <= self.batches]
        algorithm.batch_loss_log.path.write_text("\n".join(prefix) + "\n", encoding="utf-8")
        algorithm.batch_loss_log.next_id = self.batches + 1
        target = self.output / "data/training-all-branches.csv"
        with (self.folder / "failed-attempt-training.csv").open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            with target.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(r for r in reader if int(r["batch_id"]) <= self.batches)
        write_json(self.folder / "ledger.json", self.ledger)
        print(json.dumps({"resume_verified": True, "round": self.round, "next_round": 8,
                          "checkpoint_exact_replay": True, "optimizer_state_exact": True}), flush=True)

    def history(self):
        evaluations = {str(n): read_json(self.output / f"validation-{n:04d}.json") for n in (0, 3)}
        rounds = [read_json(self.output / f"round-{n:04d}.json") for n in range(1, self.round + 1)]
        return evaluations, rounds

    def finish(self, report, audit):
        after = audit.report()
        combined = combine_diagnostics(self.previous, after)
        for agent in audit.algorithm.config.agent_order:
            assert combined["gradients"][agent + "/value"]["optimizer_steps"] == 1968
            assert combined["gradients"][agent + "/actor"]["optimizer_steps"] == (240 if agent == "uc" else 744)
        assert sum(row["periods"] for row in combined["samples"] if row["phase"].startswith("round_")) == 15744
        self.ledger.update(status="completed", actual_total=dict(self.ledger["expected_actual_total"]),
                           completed_at=datetime.now(timezone.utc).isoformat(),
                           resumed_training_and_validation_seconds=report["elapsed_training_and_validation_seconds"],
                           final_checkpoint_exact_replay=report["checkpoint_exact_replay"])
        write_json(self.folder / "ledger.json", self.ledger)
        write_json(self.folder / "resumed-diagnostics.json", after)
        report.update(recovery=self.ledger, critic_optimizer_steps_effective_per_module=1920,
                      critic_optimizer_steps_executed_per_module=1968, diagnostics=combined,
                      actual_training_periods_collected=15968, actual_completed_learning_periods=15744,
                      elapsed_training_and_validation_seconds_scope="resumed rounds 8-10 and final validation",
                      optimizer_step_trace_scope=self.ledger["optimizer_step_trace_scope"])
