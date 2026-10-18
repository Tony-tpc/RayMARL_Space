"""Resumable, deterministic production-environment validation without RL training.

Every completed batch has its own atomic checkpoint. Resumption requires the
same source/specification fingerprint; a failed batch is replayed, never counted
as successful. Workers use separate engines and independent random streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "environments/sn_energy_market/src"))
sys.dont_write_bytecode = True


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint():
    paths = sorted((ROOT / "environments/sn_energy_market/src/snmarket").glob("*.py"))
    paths += sorted((ROOT / "environments/sn_energy_market/configs").glob("*.yaml"))
    paths += [Path(__file__), Path(__file__).with_name("validation_helpers.py"),
              ROOT / "environments/sn_energy_market/tests/v9_oracle.py",
              ROOT / "environments/sn_energy_market/tests/v9_checks.py",
              ROOT / "environments/sn_energy_market/docs/v9/modeling_complete_zh.md"]
    result = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return {"combined_sha256": digest, "files": result}


def episode_size(index):
    """Fixed proportions: 20/20/40/18/2 percent at N=1/2/8/32/128."""
    bucket = index % 100
    return 1 if bucket < 20 else 2 if bucket < 40 else 8 if bucket < 80 else 32 if bucket < 98 else 128


def make_tasks(args):
    tasks = []
    for kind, count, batch in (("static", args.static_cases, args.static_batch),
                               ("invalid", args.invalid_cases, args.invalid_batch),
                               ("episode", args.episodes, args.episode_batch),
                               ("long", args.long_episodes, 1),
                               ("month", args.month_episodes, 1)):
        for start in range(0, count, batch):
            tasks.append({"id": f"{kind}-{start:08d}", "kind": kind, "start": start,
                          "count": min(batch, count - start), "seed": args.seed,
                          "periods": args.periods if kind == "episode" else
                          args.long_periods if kind == "long" else
                          args.month_periods if kind == "month" else 0})
    return tasks


def run_task(task, output):
    # Import locally so the coordinator never initializes simulation/frameworks.
    from validation_helpers import execute_batch, json_safe

    started = time.perf_counter()
    try:
        report = execute_batch(task, size_function=episode_size)
        report.update(status="passed", task=task, seconds=time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 -- persist arbitrary simulation failures for replay.
        payload = getattr(exc, "validation_context", {})
        failure_path = Path(output) / "failures" / f"{task['id']}.json"
        failure = {"status": "failed", "task": task, "exception": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(),
                   "replay_context": json_safe(payload)}
        atomic_json(failure_path, failure)
        report = {"status": "failed", "task": task, "seconds": time.perf_counter() - started,
                  "failure": str(failure_path), "message": str(exc)}
    atomic_json(Path(output) / "batches" / f"{task['id']}.json", report)
    return report


def summarize(reports, planned, elapsed):
    totals = Counter()
    coverage = Counter()
    maximums = {}
    kinds = Counter()
    failures = []
    for report in reports:
        if report["status"] != "passed":
            failures.append(report)
            continue
        kinds[report["task"]["kind"]] += report["task"]["count"]
        totals.update(report.get("totals", {}))
        coverage.update(report.get("coverage", {}))
        for name, value in report.get("maximums", {}).items():
            maximums[name] = max(maximums.get(name, 0.0), value)
    return {"status": "failed" if failures else "passed" if len(reports) == planned else "running",
            "completed_batches": len(reports), "planned_batches": planned,
            "successful_cases_by_kind": dict(kinds), "totals": dict(totals),
            "coverage": dict(coverage), "maximums": maximums, "failures": failures,
            "coordinator_elapsed_seconds": elapsed,
            "scope": "Production environment and independent arithmetic oracle; no RL training or equilibrium proof"}


def acceptance_checks(result, tasks):
    counts = Counter()
    for task in tasks:
        counts[task["kind"]] += task["count"]
    full = (counts["static"] >= 100_000 and counts["invalid"] >= 10_000
            and counts["episode"] >= 10_000 and counts["long"] >= 100 and counts["month"] >= 20)
    result["full_workload"] = full
    if full:
        required = ["charge_periods", "discharge_periods", "idle_periods", "due_consumer_periods",
                    "due_over_capacity", "early_repayment", "shift_out", "negative_signed_DR",
                    "event_on", "event_off", "zero_price_event", "monthly_rollover", "true_terminal",
                    "native_shortage", "full_service", "exact_bid_ties", "snapshot_continuations",
                    "continuous_buffer_append", "size_32_episodes", "size_128_episodes",
                    "tariff_band_flat", "tariff_band_high", "tariff_band_peak", "empty_tariff_schedule",
                    "multiple_tariff_intervals", "fractional_tariff_hours", "retained_task_ids", "new_task_ids"]
        required += [f"invalid_case_{i}" for i in range(12)]
        missing = [key for key in required if not result["coverage"].get(key, 0)]
        result["coverage_gate"] = {"required": required, "missing": missing}
        if missing:
            result["status"] = "failed"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--static-cases", type=int, default=100_000)
    parser.add_argument("--invalid-cases", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--periods", type=int, default=96)
    parser.add_argument("--long-episodes", type=int, default=100)
    parser.add_argument("--long-periods", type=int, default=672)
    parser.add_argument("--month-episodes", type=int, default=20)
    parser.add_argument("--month-periods", type=int, default=3360)
    parser.add_argument("--static-batch", type=int, default=1000)
    parser.add_argument("--invalid-batch", type=int, default=250)
    parser.add_argument("--episode-batch", type=int, default=10)
    parser.add_argument("--keep-going", action="store_true", help="Finish other batches after a failure")
    parser.add_argument("--plan-only", action="store_true", help="Print planned workload without writing files")
    args = parser.parse_args()
    for name in ("static_cases", "invalid_cases", "episodes", "long_episodes", "month_episodes"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("workers", "periods", "long_periods", "month_periods", "static_batch", "invalid_batch", "episode_batch"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main():
    args = parse_args()
    if args.replay:
        failure = json.loads(args.replay.read_text(encoding="utf-8"))
        from validation_helpers import execute_batch
        print(json.dumps(execute_batch(failure["task"], size_function=episode_size), indent=2), flush=True)
        return
    tasks = make_tasks(args)
    if args.plan_only:
        print(json.dumps({"batches": len(tasks), "static_cases": args.static_cases,
                          "invalid_cases": args.invalid_cases, "episodes": args.episodes,
                          "dynamic_physical_periods": args.episodes * args.periods
                          + args.long_episodes * args.long_periods + args.month_episodes * args.month_periods,
                          "workers": args.workers}, indent=2))
        return
    sources = fingerprint()
    if args.resume:
        output = args.resume.resolve()
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["sources"] != sources:
            raise ValueError("Resume refused: implementation, oracle, runner, or specification changed")
        tasks = manifest["tasks"]
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (args.output or ROOT / "outputs/snmarket-validation" / f"v9-{stamp}").resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "batches").mkdir()
        (output / "failures").mkdir()
        atomic_json(output / "manifest.json", {"created_utc": stamp, "sources": sources,
                    "tasks": tasks, "python": sys.version, "no_training": True})
    reports, pending = [], []
    for task in tasks:
        path = output / "batches" / f"{task['id']}.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if existing and existing["status"] == "passed" and existing["task"] == task:
            reports.append(existing)
        else:
            pending.append(task)
    started = time.perf_counter()
    print(f"Validation output: {output}; cached batches: {len(reports)}; pending: {len(pending)}", flush=True)
    aborted = False
    # Only a bounded number of tasks is dispatched, permitting timely stop/restart.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        iterator = iter(pending)
        active = {}
        for _ in range(min(len(pending), args.workers * 2)):
            task = next(iterator)
            active[pool.submit(run_task, task, str(output))] = task
        last_print = 0.0
        while active:
            completed, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in completed:
                task = active.pop(future)
                try:
                    report = future.result()
                except Exception as exc:  # noqa: BLE001 -- worker failure must become a resumable report.
                    report = {"status": "failed", "task": task, "message": repr(exc),
                              "failure": "Worker/coordinator failure; batch remains replayable"}
                reports.append(report)
                aborted = aborted or (report["status"] != "passed" and not args.keep_going)
                next_task = None if aborted else next(iterator, None)
                if next_task:
                    active[pool.submit(run_task, next_task, str(output))] = next_task
            elapsed = time.perf_counter() - started
            summary = summarize(reports, len(tasks), elapsed)
            atomic_json(output / "progress.json", summary)
            if completed or elapsed - last_print >= 30:
                print(json.dumps({"elapsed_s": round(elapsed, 1), "batches": len(reports),
                                  "planned": len(tasks), "status": summary["status"],
                                  **summary["totals"]}, ensure_ascii=False), flush=True)
                last_print = elapsed
    result = summarize(reports, len(tasks), time.perf_counter() - started)
    acceptance_checks(result, tasks)
    if fingerprint() != sources:
        result["status"] = "failed"
        result["source_changed_during_run"] = True
    if aborted:
        result["status"] = "failed"
    atomic_json(output / "report.json", result)
    print(f"Report: {output / 'report.json'}", flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
