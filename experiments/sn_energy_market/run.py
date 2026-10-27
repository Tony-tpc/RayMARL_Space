"""Fixed 72-round V9 always run, gated by validation and a clean Git commit."""

import os
from pathlib import Path

BASE = Path("D:/Codex/2026-10-27/raymarl-v9-always128-72")
ROUNDS = 72


def main():
    runtime = BASE / "runtime/train"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMP=str(runtime), TEMP=str(runtime), PYTHONDONTWRITEBYTECODE="1",
                      RAY_AUTH_MODE="token", RAY_AUTH_TOKEN_PATH=str(runtime / "auth_token"),
                      RAY_USAGE_STATS_ENABLED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    import json
    import shutil
    import subprocess
    import time

    from experiments.sn_energy_market.evaluation import retained_optimizer_steps
    from experiments.sn_energy_market.run_io import (
        exclusive_run,
        now,
        read_json,
        source_manifest,
        verify_sources,
        write_json,
    )
    from experiments.sn_energy_market.runtime import (
        ROOT,
        configuration,
        digest_tree,
        initialize_fresh,
        ray_session,
    )
    from experiments.sn_energy_market.training_trace import TrainingTrace
    from experiments.sn_energy_market.validate_policy import complete_mc

    output = BASE / "training"
    output.mkdir(parents=True, exist_ok=True)

    def status(stage, completed, **extra):
        write_json(output / "status.json", {"status": stage, "completed_rounds": completed,
                   "target_rounds": ROUNDS, "pid": os.getpid(), "updated_utc": now(), **extra})

    with exclusive_run(BASE / "run.lock"):
        if (output / "status.json").exists():
            raise FileExistsError("This run already exists; do not overwrite or restart it")
        ready = read_json(BASE / "verification/ready.json")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        if not ready["passed"] or commit != ready["git_commit"] or branch != ready["branch"] or dirty:
            raise RuntimeError("Training requires the validated clean algorithm-implementation commit")
        manifest = source_manifest()
        if manifest != ready["source_sha256"]:
            raise RuntimeError("Sources changed after validation")
        for name in manifest:
            target = output / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        cfg = configuration(output)
        write_json(output / "protocol.json", {
            "git_commit": commit, "branch": branch, "source_sha256": manifest,
            "learning_contract": cfg.learning_contract(), "outer_rounds": ROUNDS,
            "initialization": "seed11_random_weights_empty_Adam_zero_PopArt_history_month_start",
            "phase_samples": 1536, "samples_per_quarter": 384, "phases_per_round": 4,
            "ppo_periods": ROUNDS * 6144, "committed_periods": ROUNDS * 3072,
            "critic_steps_executed_each": ROUNDS * 384,
            "uc_actor_steps_executed_and_retained": ROUNDS * 48,
            "consumer_actor_steps_executed_each": ROUNDS * 144,
            "consumer_actor_steps_retained_each": ROUNDS * 96,
            "candidate_evaluation": "two_candidates_times_two_complete_months_every_round",
            "final_validation": "four_complete_months_times_three_action_seeds",
            "test_loaded": False, "mc_training_labels": 0,
        })
        status("initializing", 0)
        algorithm = trace = None
        try:
            with ray_session(runtime):
                try:
                    algorithm = cfg.build_algo()
                    weights_hash = initialize_fresh(algorithm)
                    assert algorithm.outer_round == algorithm.sampled_periods == 0
                    assert algorithm.env_runner.lane_steps == [0, 0, 0, 0]
                    assert all(int(m.value.popart_updates) == 0
                               for m in algorithm.learner_group._learner.module.values())
                    write_json(output / "initialization.json", {
                        "weights_sha256": weights_hash, "empty_Adam": True,
                        "popart_updates": 0, "outer_round": 0, "lane_steps": [0, 0, 0, 0],
                    })
                    trace = TrainingTrace(algorithm, output).install()
                    for n in range(1, ROUNDS + 1):
                        verify_sources(manifest)
                        status("training", n - 1, active_round=n)
                        started = time.perf_counter()
                        algorithm.train()
                        trace.assert_budget(n)
                        assert algorithm.last_round["selected_candidate"] == 1
                        learner = algorithm.learner_group._learner
                        popart = {key: {name: float(getattr(m.value, "popart_" + name))
                                          for name in ("shift", "scale", "mass", "updates")}
                                  for key, m in learner.module.items()}
                        assert popart["uc"]["updates"] == 3 * n
                        assert all(popart[key]["updates"] == 2 * n for key in algorithm.consumer_ids)
                        record = {"round": algorithm.last_round, "popart": popart,
                                  "lanes": algorithm.env_runner.lane_cursors(),
                                  "retained_optimizer_steps": retained_optimizer_steps(learner),
                                  "seconds": time.perf_counter() - started}
                        write_json(output / f"round-{n:04d}.json", record)
                        checkpoint = algorithm.save_to_path()
                        write_json(output / "checkpoint.json", {"path": checkpoint, "outer_round": n})
                        status("training" if n < ROUNDS else "trained", n, checkpoint=checkpoint,
                               last_round_seconds=record["seconds"])
                        print(json.dumps({"completed_rounds": n, "seconds": record["seconds"]}), flush=True)
                    algorithm.export_actors(output / "actors")
                    saved = algorithm.get_state()
                    trace.close()
                    first = algorithm.env_runner.collect_phase(32)
                    algorithm.restore_from_path(checkpoint)
                    second = algorithm.env_runner.collect_phase(32)
                    assert digest_tree(first.policy_batches) == digest_tree(second.policy_batches)
                    algorithm.set_state(saved)
                    assert digest_tree(saved) == digest_tree(algorithm.get_state())
                    status("evaluating", ROUNDS, checkpoint=checkpoint)
                    evaluation = complete_mc(algorithm, output / "validation")
                    assert evaluation["periods"] == 17712 and len(evaluation["episodes"]) == 12
                    write_json(output / "result.json", {
                        "status": "completed", "completed_rounds": ROUNDS, "checkpoint": checkpoint,
                        "checkpoint_replay_exact": True, "evaluation_state_unchanged": True,
                        "ppo_periods": algorithm.sampled_periods,
                        "committed_periods": algorithm.env_runner.trajectory_periods,
                        "optimizer_steps_executed": dict(trace.steps),
                        "validation": str(output / "validation/summary.json"),
                    })
                    status("completed", ROUNDS, checkpoint=checkpoint)
                finally:
                    if trace is not None:
                        trace.close()
                    if algorithm is not None:
                        algorithm.stop()
        except BaseException as exc:
            status("failed", algorithm.outer_round if algorithm is not None else 0, error=repr(exc))
            raise


if __name__ == "__main__":
    main()
