"""Continue the completed SOC run to round 72 with two-month candidate screening."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from experiments.sn_energy_market.batch_critic_pilot import (
    REFERENCE,
    round_budget,
    run_pilot,
)
from experiments.sn_energy_market.evaluation import ROOT, cursor, source_hashes
from experiments.sn_energy_market.hybrid_storage_study import (
    StudyObserver,
    check_protocol,
    utc_now,
)
from experiments.sn_energy_market.study_recovery import combine_diagnostics, read_json
from experiments.sn_energy_market.train import write_json
from experiments.sn_energy_market.training_audit import assert_same_state

SOURCE = ROOT / "outputs/ausgrid-hybrid-storage/20261023-soc-mode-10rounds"
OUTPUT = ROOT / "outputs/ausgrid-hybrid-storage/20261023-soc-mode-72rounds"
CANDIDATE_SEEDS = (10042, 10048)  # July and January: winter and summer, complete months.


def change_candidate_seeds(algorithm, seeds):
    """Explicit experiment protocol change, after exact native checkpoint restore."""
    before = algorithm.config.learning_contract()
    assert len(set(seeds)) >= 2 and set(seeds) <= set(before["validation_seeds"])
    config = algorithm.config.copy(copy_frozen=False)
    config.validation_seeds = tuple(seeds)
    config.validate()
    after = config.learning_contract()
    assert {k for k in before if before[k] != after[k]} <= {"validation_seeds"}
    config.freeze()
    algorithm.config = config
    algorithm.env_runner.config = config
    # RLlib Learner keeps its own mutable config for policies_to_train on restore.
    # Preserve that lifecycle and the component's other settings.
    def update_component(component):
        local = component.config.copy(copy_frozen=False)
        local.validation_seeds = tuple(seeds)
        if component.config._is_frozen:
            local.freeze()
        component.config = local

    update_component(algorithm.learner_group)
    for result in algorithm.learner_group.foreach_learner(update_component):
        result.get()
    return after


class StudyContinuation:
    """Reuse the runner's restore interface without mixing continuation with failure recovery."""

    def __init__(self, output, source, *, candidate_seeds=CANDIDATE_SEEDS):
        self.output, self.source = Path(output).resolve(), Path(source).resolve()
        self.folder = self.output
        self.previous = read_json(self.source / "report.json")
        self.round = self.last_validated_round = self.previous["outer_rounds"]
        self.batches = self.previous["logged_batches"]
        self.checkpoint = self.source / "checkpoint"
        self.candidate_seeds = tuple(candidate_seeds)

    def verify(self, cfg, outer_rounds):
        assert self.previous["status"] == "completed" and self.previous["checkpoint_exact_replay"]
        assert self.round == 10 and outer_rounds == 72
        assert cfg.learning_contract() == self.previous["learning_contract"]
        assert read_json(self.checkpoint / "learning_contract.json") == cfg.learning_contract()
        assert source_hashes("environments/sn_energy_market/src") == self.previous["environment_source_sha256"]
        for relative, digest in source_hashes("RLAlgo/src/raymarl/algorithms/stackelberg").items():
            assert self.previous["source_sha256"][relative] == digest, relative
        assert set(self.candidate_seeds) <= set(cfg.validation_seeds) and len(self.candidate_seeds) >= 2
        self.output.mkdir(parents=True, exist_ok=False)
        # Evidence from completed rounds is copied once; no previous run is overwritten.
        shutil.copytree(self.source / "data", self.output / "data")
        for path in self.source.glob("round-*.json"):
            shutil.copy2(path, self.output / path.name)
        for path in self.source.glob("validation-*.json"):
            shutil.copy2(path, self.output / path.name)
        self.ledger = {
            "status": "prepared", "source": str(self.source), "checkpoint": str(self.checkpoint),
            "source_round": self.round, "target_round": outer_rounds,
            "additional_rounds": outer_rounds - self.round,
            "additional_budget": {k: v * (outer_rounds - self.round) for k, v in round_budget(cfg).items()},
            "candidate_seeds_before": list(cfg.validation_seeds),
            "candidate_seeds_after": list(self.candidate_seeds), "full_validation_rounds": [30, 72],
            "full_period_checks": False, "created_at": utc_now(),
        }
        write_json(self.folder / "ledger.json", self.ledger)

    def restore(self, algorithm):
        algorithm.restore_from_path(str(self.checkpoint))
        assert algorithm.outer_round == algorithm.training_iteration == self.round
        assert algorithm.completed_updates == self.batches
        assert algorithm.sampled_periods == self.previous["ppo_sampled_periods"]
        assert cursor(algorithm.env_runner) == self.previous["end"]
        saved = pickle.loads((self.checkpoint / "algorithm_state.pkl").read_bytes())
        learner_state = pickle.loads((self.checkpoint / "learner_group/learner/state.pkl").read_bytes())
        learner = algorithm.learner_group._learner
        assert_same_state(learner_state["optimizer"], learner.get_state()["optimizer"])
        for agent in algorithm.config.agent_order:
            path = self.checkpoint / f"learner_group/learner/rl_module/{agent}/module_state.pkl"
            assert_same_state(pickle.loads(path.read_bytes()), learner.module[agent].get_state())
        assert_same_state(saved["raymarl_rng"], algorithm.get_state()["raymarl_rng"])
        expected = algorithm.env_runner.collect(8)
        algorithm.restore_from_path(str(self.checkpoint))
        actual = algorithm.env_runner.collect(8)
        for key, data in expected.policy_batches.items():
            for name, values in data.items():
                np.testing.assert_array_equal(values, actual[key][name])
        algorithm.restore_from_path(str(self.checkpoint))
        before = algorithm.get_state()
        contract = change_candidate_seeds(algorithm, self.candidate_seeds)
        after = algorithm.get_state()
        for key in before:
            if key != "learning_contract":
                assert_same_state(before[key], after[key])
        assert Path(algorithm.logdir).resolve().is_relative_to(self.output)
        algorithm.batch_loss_log.path.write_bytes((self.source / "batch_losses.jsonl").read_bytes())
        algorithm.batch_loss_log.next_id = self.batches + 1
        self.ledger.update(status="restored", native_weights_and_adam_exact=True,
                           checkpoint_exact_replay=True, replay_diagnostic_periods=16,
                           replay_diagnostics_rewound=True, configuration_change_only="validation_seeds",
                           learning_contract=contract, restored_at=utc_now())
        write_json(self.folder / "ledger.json", self.ledger)
        print(json.dumps({"continuation_verified": True, "from_round": self.round,
                          "candidate_seeds": self.candidate_seeds, "checkpoint_exact_replay": True}), flush=True)

    def history(self):
        return ({str(n): read_json(self.source / f"validation-{n:04d}.json")
                 for n in (0, 3, 10)},
                [read_json(self.source / f"round-{n:04d}.json") for n in range(1, self.round + 1)])

    def finish(self, report, audit):
        combined = combine_diagnostics(self.previous["diagnostics"], audit.report())
        rounds = report["outer_rounds"]
        for agent in audit.algorithm.config.agent_order:
            assert combined["gradients"][agent + "/value"]["optimizer_steps"] == 192 * rounds
            assert combined["gradients"][agent + "/actor"]["optimizer_steps"] == (24 if agent == "uc" else 72) * rounds
        assert sum(x["periods"] for x in combined["samples"] if x["phase"].startswith("round_")) == 1536 * rounds
        self.ledger.update(status="completed", completed_at=utc_now(),
                           final_checkpoint_exact_replay=report["checkpoint_exact_replay"])
        write_json(self.folder / "ledger.json", self.ledger)
        report.update(continuation=self.ledger, diagnostics=combined,
                      elapsed_training_and_validation_seconds_scope="additional rounds 11-72 and validations 30/72",
                      optimizer_step_trace_scope="actual previous 1-10 records plus actual continued 11-72 records")
        # The original complete trace exists; concatenate actual records, never inferred updates.
        audit.optimizer_steps[:0] = read_json(self.source / "optimizer_steps.json")


def main():
    continuation = StudyContinuation(OUTPUT, SOURCE)
    protocol = check_protocol(OUTPUT, outer_rounds=72, previous=SOURCE)
    protocol.update(validation_rounds=[30, 72], inherited_validation_rounds=[0, 3, 10],
                    candidate_validation_seeds=list(CANDIDATE_SEEDS),
                    changes_from_previous_ten_rounds=["validation_seeds"],
                    continuation_source=str(SOURCE), full_period_checks=False)
    observer = StudyObserver(OUTPUT, protocol, recovery=continuation)
    run_pilot(OUTPUT, REFERENCE, outer_rounds=72, observer=observer, checkpoint_each_round=True,
              recovery=continuation, evaluation_rounds=(30, 72), full_period_checks=False)


if __name__ == "__main__":
    main()
