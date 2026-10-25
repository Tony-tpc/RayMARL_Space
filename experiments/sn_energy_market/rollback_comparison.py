"""Eight rounds from D24: paired-gain selection versus always retaining the candidate."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.sn_energy_market.sampling_gate import ROOT
from experiments.sn_energy_market.sampling_gate.audit import ExperimentAudit
from experiments.sn_energy_market.sampling_gate.evaluate import (
    common_state_modes,
    complete_mc,
)
from experiments.sn_energy_market.sampling_gate.io import (
    source_manifest,
    verify_sources,
    write_json,
)
from experiments.sn_energy_market.sampling_gate.runtime import (
    ExperimentStackelberg,
    configuration,
    digest_tree,
    ray_session,
)

SOURCE = Path("D:/Codex/2026-10-25/critic-actor-d-576-uc-actor/work/rotation-promotion")
OUTPUT = Path("D:/Codex/2026-10-25/critic-actor-d-576-uc-actor/outputs/rotation-rollback-8rounds")
START, END = 24, 32


class RollbackComparison(ExperimentStackelberg):
    def _select_candidate(self, evaluated):
        selected, gains = super()._select_candidate(evaluated)
        self.gate_selected = selected
        if self.config.always_keep_candidate:
            assert len(evaluated) == 2
            selected = 1
        return selected, gains


def study_config(group):
    cfg = configuration(OUTPUT / group, "D")
    cfg.algo_class = RollbackComparison
    cfg.initialization = "D24_full_learner_adam_sampling_state"
    cfg.always_keep_candidate = group == "always"
    return cfg


def import_d_state(algorithm, *, budget_changes=None):
    """Explicit one-time schema transfer; all scientific and optimizer settings must match."""
    with (SOURCE / "D-state.pkl").open("rb") as stream:
        state = pickle.load(stream)
    old, new = state["learning_contract"], algorithm.config.learning_contract()
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    # The factory gained only a grouping property. Do not bypass its other hashes,
    # environment settings, monthly windows, household identities or dataset hashes.
    old_scenario, new_scenario = copy.deepcopy(old["scenario"]), copy.deepcopy(new["scenario"])
    old_factory = old_scenario["source_sha256"].pop("profile_dataset.py")
    new_factory = new_scenario["source_sha256"].pop("profile_dataset.py")
    assert old_scenario == new_scenario, "Environment/data contract changed"
    permitted = {"phase_sampling": "single_stream", **(budget_changes or {})}
    assert set(permitted) <= {"phase_sampling", "inner_updates", "response_updates", "uc_updates_per_candidate",
                              "critic_epochs", "num_epochs"}
    assert all(new[key] == value for key, value in permitted.items())
    assert set(changed) <= {"train_seed_groups", "sampling_protocol", "experiment", "scenario", *permitted}, changed
    sampling = state["env_runner"]["sampling"]
    streams = sampling.pop("experiment_streams")
    assert streams.pop("order") == "rotation"
    sampling["streams"] = dict(streams, seed_groups=algorithm.config.train_seed_groups)
    state["learning_contract"] = new
    assert state["raymarl_algorithm"] == "ExperimentStackelberg"
    state["raymarl_algorithm"] = type(algorithm).__name__
    sampling_before = copy.deepcopy(sampling)
    algorithm.set_state(state)
    assert algorithm.outer_round == START
    source = json.loads((SOURCE / "D-export.json").read_text(encoding="utf-8"))
    learner_sha256 = digest_tree(algorithm.learner_group.get_state())
    assert learner_sha256 == source["learner_sha256"], "D parameters or Adam state changed"
    restored = algorithm.env_runner.get_state(components="sampling")["sampling"]
    assert digest_tree({k: v for k, v in restored.items() if k != "streams"}) == digest_tree(
        {k: v for k, v in sampling_before.items() if k != "streams"})
    for lane, saved in enumerate(sampling_before["streams"]["bank"]):
        if lane != restored["streams"]["active_lane"]:
            assert digest_tree(saved) == digest_tree(restored["streams"]["bank"][lane])
    return {"changed_contract_fields": changed, "learner_sha256": learner_sha256,
            "factory_source_sha256": {"before": old_factory, "after": new_factory},
            "environment_data_contract_equal": True, "source": source}


def assert_batches_equal(left, right):
    assert left.count == right.count and left.policy_batches.keys() == right.policy_batches.keys()
    for key, data in left.policy_batches.items():
        assert data.keys() == right[key].keys()
        for name, value in data.items():
            np.testing.assert_array_equal(value, right[key][name])


def preflight():
    """Compare the promoted runner against batches exported before the source edit."""
    with (SOURCE / "D-replay.pkl").open("rb") as stream:
        expected = pickle.load(stream)
    with ray_session(Path("D:/Codex/rotation-runtime/preflight")):
        algorithm = study_config("gate").build_algo()
        try:
            result = import_d_state(algorithm)
            for record in expected:
                algorithm.env_runner.begin_round(record["round"])
                assert_batches_equal(record["batch"], algorithm.env_runner.collect(8))
                assert algorithm.env_runner.lane_cursors() == record["lanes"]
            result.update(passed=True, four_stream_replay_exact=True, optimizer_steps=0)
            write_json(OUTPUT / "preflight.json", result)
            manifest = source_manifest()
            for name in manifest:
                target = OUTPUT / "sources" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            write_json(OUTPUT / "protocol.json", {
                "source_sha256": manifest, "start_round": START, "end_round": END,
                "groups": {"gate": "paired UC gain, z=1", "always": "retain candidate 1"},
                "train_seed_groups": algorithm.config.train_seed_groups,
                "same_D_parameters_Adam_and_sampling": True,
                "per_arm": {"rounds": 8, "learning_batches": 32, "ppo_periods": 12288,
                            "main_periods": 6144, "uc_actor_steps_executed": 192,
                            "critic_steps_executed_each": 1536},
                "evaluation": "same 24 physical states; 4 paired continuations per mode plus policy control; complete month MC on four validation months and three action seeds",
                "test_loaded": False, "extra_MC_training_labels": 0,
            })
            print(json.dumps(result), flush=True)
        finally:
            algorithm.stop()


def train(group):
    output = OUTPUT / group
    assert not (output / "status.json").exists(), "Existing study evidence must not be overwritten"
    manifest = source_manifest()
    write_json(output / "sources.json", manifest)
    with ray_session(Path("D:/Codex/rotation-runtime") / group):
        algorithm = study_config(group).build_algo()
        audit = None
        try:
            migration = import_d_state(algorithm)
            write_json(output / "initialization.json", migration)
            audit = ExperimentAudit(algorithm, output).install()
            audit.assert_budget(START)
            for n in range(START + 1, END + 1):
                verify_sources(manifest)
                write_json(output / "status.json", {"status": "training", "group": group,
                           "completed_rounds": n - 1 - START, "active_round": n - START})
                started = time.perf_counter()
                algorithm.train()
                audit.assert_budget(n)
                record = {"round": algorithm.last_round, "gate_selected": algorithm.gate_selected,
                          "lanes": algorithm.env_runner.lane_cursors(),
                          "seconds": time.perf_counter() - started}
                write_json(output / f"round-{n:04d}.json", record)
                saved = algorithm.save_to_path()
                write_json(output / "checkpoint.json", {"path": saved, "outer_round": n})
                print(json.dumps({"group": group, "round": n - START,
                                 "accepted": algorithm.last_round["candidate_accepted"],
                                 "gate_selected": algorithm.gate_selected,
                                 "seconds": record["seconds"]}), flush=True)
            algorithm.export_actors(output / "actors")
            write_json(output / "status.json", {"status": "trained", "group": group,
                                               "completed_rounds": END - START})
        except BaseException:
            write_json(output / "status.json", {"status": "failed", "group": group,
                       "completed_rounds": algorithm.outer_round - START})
            raise
        finally:
            if audit is not None:
                audit.close()
            algorithm.stop()


def evaluate(group):
    output = OUTPUT / group
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["outer_round"] == END
    with ray_session(Path("D:/Codex/rotation-runtime") / f"evaluate-{group}"):
        algorithm = study_config(group).build_algo()
        try:
            algorithm.restore_from_path(checkpoint["path"])
            saved = copy.deepcopy(algorithm.get_state())
            expected = algorithm.env_runner.collect(8)
            algorithm.restore_from_path(checkpoint["path"])
            assert_batches_equal(expected, algorithm.env_runner.collect(8))
            algorithm.restore_from_path(checkpoint["path"])
            assert digest_tree(algorithm.get_state()) == digest_tree(saved)
            for phase, function, folder in (("full_mc", complete_mc, "validation"),
                                             ("mode_mc", common_state_modes, "mode-evaluation")):
                write_json(output / "status.json", {"status": "evaluating", "phase": phase,
                           "group": group, "completed_rounds": END - START})
                function(algorithm, output / folder)
            write_json(output / "status.json", {"status": "completed", "group": group,
                       "completed_rounds": END - START, "checkpoint_replay_exact": True})
        finally:
            algorithm.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "evaluate", "all"))
    parser.add_argument("--group", choices=("gate", "always"))
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
        return
    assert json.loads((OUTPUT / "preflight.json").read_text())["passed"]
    verify_sources(json.loads((OUTPUT / "protocol.json").read_text())["source_sha256"])
    groups = (args.group,) if args.group else ("gate", "always")
    if args.stage in ("train", "all"):
        for group in groups:
            train(group)
    if args.stage in ("evaluate", "all"):
        for group in groups:
            evaluate(group)


if __name__ == "__main__":
    main()
