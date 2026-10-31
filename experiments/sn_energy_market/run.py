"""Fresh training, gated by validation and a clean Git commit; never auto-resume."""

import os
from pathlib import Path


def main(base, rounds):
    base = Path(base).resolve()
    if rounds < 1:
        raise ValueError("rounds must be positive")
    runtime = base / "runtime/train"
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
        EVALUATION_SEEDS,
        MONTH_SEEDS,
        ROOT,
        configuration,
        digest_tree,
        initialize_fresh,
        ray_session,
    )
    from experiments.sn_energy_market.training_trace import TrainingTrace
    from experiments.sn_energy_market.validate_policy import complete_mc

    output = base / "training"
    output.mkdir(parents=True, exist_ok=True)

    def status(stage, completed, **extra):
        write_json(output / "status.json", {"status": stage, "completed_rounds": completed,
                   "target_rounds": rounds, "pid": os.getpid(), "updated_utc": now(), **extra})

    with exclusive_run(base / "run.lock"):
        if (output / "status.json").exists():
            raise FileExistsError("This run already exists; do not overwrite or restart it")
        ready = read_json(base / "verification/ready.json")
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
        windows = cfg.scenario_factory.get_contract()["seed_windows"]
        validation_periods = sum(
            windows[str(seed)]["stop"] - windows[str(seed)]["start"] for seed in MONTH_SEEDS
        ) * len(EVALUATION_SEEDS)
        write_json(output / "protocol.json", {
            "git_commit": commit, "branch": branch, "source_sha256": manifest,
            "learning_contract": cfg.learning_contract(), "outer_rounds": rounds,
            "initialization": "seed11_random_weights_empty_Adam_zero_PopArt_history_month_start",
            "phase_samples": 1536, "samples_per_quarter": 384, "phases_per_round": 4,
            "ppo_periods": rounds * 6144, "committed_periods": rounds * 3072,
            "critic_steps_executed_each": rounds * 384,
            "uc_actor_steps_executed_and_retained": rounds * 48,
            "consumer_actor_steps_executed_each": rounds * 144,
            "consumer_actor_steps_retained_each": rounds * 96,
            "candidate_evaluation": "two_candidates_times_two_complete_months_every_round",
            "final_validation": "twelve_complete_months_times_three_action_seeds",
            "validation_periods": validation_periods,
            "test_loaded": False, "validation_labels_used_for_training": 0,
            "mode_credit": cfg.mode_credit,
            "mode_head_update_calls": rounds,
            "mode_probe_contexts_per_round": 24,
            "mode_simulated_periods_upper_bound_per_round": 29184,
            "uc_mode_head_Adam_updates": 0,
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
                    for n in range(1, rounds + 1):
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
                        status("training" if n < rounds else "trained", n, checkpoint=checkpoint,
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
                    status("evaluating", rounds, checkpoint=checkpoint)
                    evaluation = complete_mc(
                        algorithm, output / "validation",
                        progress=lambda **progress: status("evaluating", rounds, checkpoint=checkpoint, **progress),
                    )
                    assert evaluation["periods"] == validation_periods and len(evaluation["episodes"]) == 36
                    write_json(output / "result.json", {
                        "status": "completed", "completed_rounds": rounds, "checkpoint": checkpoint,
                        "checkpoint_replay_exact": True, "evaluation_state_unchanged": True,
                        "ppo_periods": algorithm.sampled_periods,
                        "mode_simulated_periods": algorithm.mode_sampled_periods,
                        "committed_periods": algorithm.env_runner.trajectory_periods,
                        "optimizer_steps_executed": dict(trace.steps),
                        "validation": str(output / "validation/summary.json"),
                    })
                    status("completed", rounds, checkpoint=checkpoint)
                finally:
                    if trace is not None:
                        trace.close()
                    if algorithm is not None:
                        algorithm.stop()
        except BaseException as exc:
            status("failed", algorithm.outer_round if algorithm is not None else 0, error=repr(exc))
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args()
    main(args.output, args.rounds)
