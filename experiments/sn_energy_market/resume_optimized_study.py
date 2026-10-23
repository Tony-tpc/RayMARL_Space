"""Resume the saved round boundary after verified execution-only changes."""
import csv
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from experiments.sn_energy_market.batch_critic_pilot import REFERENCE, run_pilot
from experiments.sn_energy_market.evaluation import ROOT, cursor, source_hashes
from experiments.sn_energy_market.hybrid_storage_study import StudyObserver, utc_now
from experiments.sn_energy_market.study_recovery import read_json
from experiments.sn_energy_market.train import write_json
from experiments.sn_energy_market.training_audit import assert_same_state

SOURCE = ROOT / "outputs/ausgrid-hybrid-storage/20261023-soc-mode-72rounds"
OUTPUT = ROOT / "outputs/ausgrid-hybrid-storage/20261023-soc-mode-72rounds-optimized"
PREFLIGHT = ROOT / "outputs/diagnostics/optimization-parity-20261023/report.json"


class PerformanceRecovery:
    def __init__(self, output=OUTPUT, source=SOURCE):
        self.output, self.source = Path(output), Path(source)
        self.folder = self.output
        stop = read_json(self.source / "optimization-stop.json")
        self.round = int(stop["stopped_after_round"])
        self.batches, self.last_validated_round = self.round * 4, 10
        self.checkpoint = Path(stop["checkpoint"])

    def configure(self, cfg):
        contract = read_json(self.checkpoint / "learning_contract.json")
        cfg.validation_seeds = tuple(contract["validation_seeds"])
        return cfg

    def verify(self, cfg, outer_rounds):
        assert outer_rounds == 72 and 0 < self.round < outer_rounds
        assert read_json(self.checkpoint / "learning_contract.json") == cfg.learning_contract()
        preflight = read_json(PREFLIGHT)
        assert preflight["passed"] and preflight["checkpoint"] == str(self.checkpoint)
        for name, digest in preflight["optimized_source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        old = read_json(self.source / "source_manifest.json")
        for name, digest in source_hashes("environments/sn_energy_market/src").items():
            assert old[name] == digest, name
        self.output.mkdir(parents=True, exist_ok=False)
        shutil.copytree(self.checkpoint, self.output / "resume-input")
        self.checkpoint = self.output / "resume-input"
        shutil.copytree(self.source / "data", self.output / "data")
        for pattern in ("round-*.json", "validation-*.json"):
            for path in self.source.glob(pattern):
                shutil.copy2(path, self.output / path.name)
        # Keep only completed-round trajectory rows in the new logical run.
        source_csv = self.source / "data/training-all-branches.csv"
        with source_csv.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            with (self.output / "data/training-all-branches.csv").open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(row for row in reader if int(row["batch_id"]) <= self.batches)
        self.ledger = {"status": "prepared", "source": str(self.source), "checkpoint": str(self.checkpoint),
                       "resume_after_round": self.round, "target_round": outer_rounds,
                       "baseline_git": "58d8cd8a667c686245c6c58dde98e34992036e09",
                       "learning_contract_unchanged": True, "preflight": preflight,
                       "created_at": utc_now(), "full_validation_rounds": [30, 72]}
        write_json(self.folder / "ledger.json", self.ledger)

    def restore(self, algorithm):
        algorithm.restore_from_path(str(self.checkpoint))
        assert algorithm.outer_round == algorithm.training_iteration == self.round
        assert algorithm.completed_updates == self.batches
        assert algorithm.sampled_periods == self.round * 1536
        assert cursor(algorithm.env_runner) == read_json(self.source / f"round-{self.round:04d}.json")["end"]
        saved = pickle.loads((self.checkpoint / "algorithm_state.pkl").read_bytes())
        learner = algorithm.learner_group._learner
        state = pickle.loads((self.checkpoint / "learner_group/learner/state.pkl").read_bytes())
        assert_same_state(state["optimizer"], learner.get_state()["optimizer"])
        for agent in algorithm.config.agent_order:
            path = self.checkpoint / f"learner_group/learner/rl_module/{agent}/module_state.pkl"
            assert_same_state(pickle.loads(path.read_bytes()), learner.module[agent].get_state())
        assert_same_state(saved["raymarl_rng"], algorithm.get_state()["raymarl_rng"])
        expected = algorithm.env_runner.collect(8)
        algorithm.restore_from_path(str(self.checkpoint))
        actual = algorithm.env_runner.collect(8)
        for key, columns in expected.policy_batches.items():
            for name, values in columns.items():
                np.testing.assert_array_equal(values, actual[key][name])
        algorithm.restore_from_path(str(self.checkpoint))
        old_log = next((self.source / "raymarl").glob("*/batch_losses.jsonl"))
        lines = [line for line in old_log.read_text(encoding="utf-8").splitlines()
                 if json.loads(line)["batch_id"] <= self.batches]
        algorithm.batch_loss_log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        algorithm.batch_loss_log.next_id = self.batches + 1
        self.ledger.update(status="restored", native_weights_and_adam_exact=True,
                           checkpoint_exact_replay=True, replay_diagnostic_periods=16,
                           replay_diagnostics_rewound=True, restored_at=utc_now())
        write_json(self.folder / "ledger.json", self.ledger)
        print(json.dumps({"resume_verified": True, "round": self.round,
                          "checkpoint_exact_replay": True}), flush=True)

    def history(self):
        return ({str(n): read_json(self.source / f"validation-{n:04d}.json") for n in (0, 3, 10)},
                [read_json(self.source / f"round-{n:04d}.json") for n in range(1, self.round + 1)])

    def finish(self, report, audit):
        self.ledger.update(status="completed", completed_at=utc_now(),
                           final_checkpoint_exact_replay=report["checkpoint_exact_replay"])
        write_json(self.folder / "ledger.json", self.ledger)
        report.update(performance_resume=self.ledger,
                      diagnostics_scope=f"resumed rounds {self.round + 1}-72; prior evidence retained at source",
                      optimizer_step_trace_scope=f"actual resumed rounds {self.round + 1}-72 only",
                      elapsed_training_and_validation_seconds_scope=f"resumed rounds {self.round + 1}-72 and validations 30/72")


def main():
    recovery = PerformanceRecovery()
    protocol = read_json(SOURCE / "protocol.json")
    protocol.update(optimization_baseline_git="58d8cd8", resumed_from_round=recovery.round,
                    execution_only_optimization=True)
    observer = StudyObserver(OUTPUT, protocol, recovery=recovery)
    run_pilot(OUTPUT, REFERENCE, outer_rounds=72, observer=observer, checkpoint_each_round=True,
              recovery=recovery, evaluation_rounds=(30, 72), full_period_checks=False)


if __name__ == "__main__":
    main()
