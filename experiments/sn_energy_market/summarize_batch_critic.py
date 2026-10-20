"""Compare saved fixed-budget runs and render actual batch losses, without training."""

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.sn_energy_market.plot_batch_losses import load_batches, plot_batches


def performance(evaluation):
    result = {}
    for mode in ("sampled", "mean"):
        rows = evaluation["performance"][mode]["rows"]
        lost = sum(row["metrics"]["native_lost_kwh"] for row in rows)
        requested = sum(row["metrics"]["native_requested_kwh"] for row in rows)
        result[mode] = {
            "service_rate_pct": 100 * (1 - lost / requested),
            "native_lost_kwh_total": lost,
            "native_requested_kwh_total": requested,
            "uc_cash_aud_per_48h": mean(row["metrics"]["uc_cash"] for row in rows),
            "uc_book_profit_aud_per_48h": mean(row["metrics"]["uc_book_profit"] for row in rows),
            "uc_discounted_return": mean(row["returns"]["uc"] for row in rows),
        }
    return result


def compare(reference, current):
    before, after = reference["final"], current["final"]
    performance_before, performance_after = performance(before), performance(after)
    critic = {}
    for key, strata in after["critic"]["modules"].items():
        critic[key] = {}
        for name, values in strata.items():
            old = before["critic"]["modules"][key][name]
            assert values["samples"] == old["samples"]
            critic[key][name] = {
                "previous": old, "current": values,
                "rmse_delta": values["rmse"] - old["rmse"],
                "rmse_change_pct": 100 * (values["rmse"] / old["rmse"] - 1),
            }
    consumers = [key for key in critic if key != "uc"]
    return {
        "performance": {
            mode: {"previous": performance_before[mode], "current": performance_after[mode],
                   "delta": {key: value - performance_before[mode][key]
                             for key, value in performance_after[mode].items()}}
            for mode in performance_after
        },
        "critic": critic,
        "consumer_mean_rmse": {
            stratum: {
                side: mean(critic[key][stratum][side]["rmse"] for key in consumers)
                for side in ("previous", "current")
            } for stratum in critic["uc"]
        },
        "consumers_with_lower_full_month_rmse": sum(critic[key]["all"]["rmse_delta"] < 0 for key in consumers),
    }


def full_month_performance(evaluation):
    episodes = evaluation["critic"]["episodes"]
    periods = sum(row["periods"] for row in episodes)
    totals = {key: sum(row["metrics"][key] for row in episodes) for key in episodes[0]["metrics"]}
    return {
        "periods": periods,
        "service_rate_pct": 100 * (1 - totals["native_lost_kwh"] / totals["native_requested_kwh"]),
        "lost_kwh": totals["native_lost_kwh"],
        "requested_kwh": totals["native_requested_kwh"],
        "cash_aud_total": totals["uc_cash"],
        "cash_aud_per_day": totals["uc_cash"] * 48 / periods,
        "book_profit_aud_per_day": totals["uc_book_profit"] * 48 / periods,
    }


def consumer_curves(rows):
    """Equal-sized batches: mean scaled loss and pooled raw-unit RMSE over consumers."""
    result = []
    label = "consumers_16_mean_loss_pooled_RMSE"
    for source in rows:
        row = copy.deepcopy(source)
        keys = [key for key in row["modules"] if key != "uc"]
        values = [row["modules"][key] for key in keys]
        combined = {}
        for name in values[0]:
            if not name.startswith(("critic_loss", "critic_rmse", "policy_loss", "value_scale")):
                continue
            data = [value[name] for value in values]
            if any(value is None for value in data):
                assert all(value is None for value in data)
                combined[name] = None
            elif name.startswith("critic_rmse"):
                combined[name] = math.sqrt(mean(value ** 2 for value in data))
            else:
                combined[name] = mean(data)
        roles = row["retained_updates"]
        assert all(roles[key] == roles[keys[0]] for key in keys)
        row["modules"] = {label: combined}
        row["retained_updates"] = {label: roles[keys[0]]}
        result.append(row)
    return result


def summarize(report_path, output):
    current = json.loads(report_path.read_text(encoding="utf-8"))
    reference_path = ROOT / current["reference_report"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    log_path = ROOT / current["batch_loss_log"]
    rows = load_batches(log_path)
    assert current["status"] == "completed"
    assert performance(current["initial"]) == performance(reference["initial"])
    assert current["initial"]["critic"]["periods"] == reference["initial"]["critic"]["periods"]
    assert current["ppo_sampled_periods"] == reference["ppo_sampled_periods"] == 1536
    assert current["critic_auxiliary_periods"] == reference["critic_auxiliary_periods"] == 0
    assert not current["test_loaded"] and not current["test_evaluated"]
    assert [row["batch_id"] for row in rows] == list(range(1, 25))
    assert all(row["branch_status"] != "pending" and row["samples"] == 64 for row in rows)
    assert hashlib.sha256(log_path.read_bytes()).hexdigest() == current["batch_loss_sha256"]
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == current["reference_sha256"]
    full_reference_path = ROOT / current["full_month_reference"]
    assert hashlib.sha256(full_reference_path.read_bytes()).hexdigest() == current["full_month_reference_sha256"]
    full_reference = json.loads(full_reference_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "completed", "test_loaded": False, "test_evaluated": False,
        "report": str(report_path.relative_to(ROOT)),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "reference": current["reference_report"], "reference_sha256": current["reference_sha256"],
        "batch_loss_sha256": current["batch_loss_sha256"],
        "budget": {"outer_rounds": 3, "batches": 24, "batch_steps": 64, "ppo_periods": 1536,
                   "main_periods": current["main_advanced_periods"],
                   "critic_epochs": 4, "actor_epochs": 2, "minibatch_size": 32,
                   "critic_steps_executed_per_module": 192, "auxiliary_periods": 0},
        "learning_contract": current["learning_contract"],
        "start": current["start"], "end": current["end"],
        "selected_candidates": [row["round"]["selected_candidate"] for row in current["rounds"]],
        "initial_actor_performance_matches_reference": True,
        "initial_critic": current["initial"]["critic"]["modules"],
        "initial_performance": performance(current["initial"]),
        "comparison": compare(reference, current),
        "full_month_reference": current["full_month_reference"],
        "full_month_reference_sha256": current["full_month_reference_sha256"],
        "full_month_performance": {
            "previous": full_month_performance(full_reference),
            "initial": full_month_performance(current["initial"]),
            "current": full_month_performance(current["final"]),
        },
        "candidate_gains": [
            {"round": row["round"]["outer_round"],
             "selected": row["round"]["selected_candidate"],
             "candidates": [
                 {"index": c["index"], "paired_uc_gain": c["paired_uc_gain"]}
                 for c in row["round"]["candidates"]
             ]}
            for row in current["rounds"]
        ],
        "validation_episodes": current["final"]["critic"]["episodes"],
        "checkpoint_exact_replay": current["checkpoint_exact_replay"],
        "retained_optimizer_steps": current["retained_optimizer_steps"],
        "environment_source_unchanged": current["environment_source_unchanged"],
        "environment_source_sha256": current["environment_source_sha256"],
        "source_sha256": current["source_sha256"],
        "audit_maximums": current["diagnostics"]["maximums"],
        "audit_coverage": current["diagnostics"]["coverage"],
        "optimizer_audit": current["diagnostics"]["gradients"],
        "sampled_dates": current["diagnostics"]["sampled_dates"],
        "batch_fit": {
            key: {"improved_batches": sum(row["modules"][key]["critic_loss_after"]
                                           < row["modules"][key]["critic_loss_before"] for row in rows),
                  "rolled_back_batches": [row["batch_id"] for row in rows
                                          if not row["retained_updates"][key]["critic"]]}
            for key in rows[0]["modules"]
        },
    }
    (output / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "batch_losses.jsonl").write_bytes(log_path.read_bytes())
    flat = []
    for row in rows:
        for key, values in row["modules"].items():
            flat.append({"batch_id": row["batch_id"], "outer_round": row["outer_round"],
                         "phase": row["phase"], "candidate_index": row["candidate_index"],
                         "module": key, "branch_status": row["branch_status"],
                         "critic_retained": row["retained_updates"][key]["critic"],
                         "actor_retained": row["retained_updates"][key]["actor"],
                         **{k: v for k, v in values.items() if isinstance(v, (float, int)) or v is None}})
    with (output / "batch_losses.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    uc_rows = copy.deepcopy(rows)
    for row in uc_rows:
        row["modules"] = {"uc": row["modules"]["uc"]}
        row["retained_updates"] = {"uc": row["retained_updates"]["uc"]}
    plot_batches(uc_rows, output / "plots")
    plot_batches(consumer_curves(rows), output / "plots")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.report.resolve(), args.output)
    print(json.dumps({"performance": summary["comparison"]["performance"],
                      "uc_rmse": summary["comparison"]["critic"]["uc"],
                      "selected_candidates": summary["selected_candidates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
