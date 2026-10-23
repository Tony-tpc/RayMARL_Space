"""Render the completed frozen-critic experiment without importing training code."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"mlp": "#475569", "time": "#0284c7", "gru": "#d97706"}
LABELS = {"mlp": "Current MLP", "time": "MLP + public time", "gru": "UC GRU"}
GROUPS = ("all", "early_1_10", "middle_11_20", "late_21_end")
DATES = ("Whole month", "Days 1-10", "Days 11-20", "Days 21-end")


def save(figure, output, stem):
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=170, facecolor="white")
    plt.close(figure)


def export_dates(run, output, report):
    """Keep every calendar date available alongside the three date bands."""
    manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
    episodes = {}
    for entry in manifest:
        if entry["split"] == "validation":
            with np.load(run / "data" / entry["file"], allow_pickle=False) as data:
                episodes[str(entry["seed"])] = (np.asarray([str(t)[:10] for t in data["timestamps"]]),
                                                data["targets"].copy())
    rows = []
    for result in report["results"]:
        kind, seed = result["kind"], result["seed"]
        with np.load(run / f"{kind}-{seed}" / "validation_predictions.npz", allow_pickle=False) as saved:
            for episode, (dates, target) in episodes.items():
                for date in np.unique(dates):
                    mask = dates == date
                    error = saved[episode][mask].astype(np.float64) - target[mask]
                    variance = np.var(target[mask])
                    rows.append({"kind": kind, "seed": seed, "date": date, "samples": int(mask.sum()),
                                 "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean()),
                                 "explained_variance": float(1 - np.var(error) / variance) if variance > 0 else 0.0})
    with (output / "per-date-metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for kind in COLORS:
        for date in sorted({row["date"] for row in rows}):
            peers = [row for row in rows if row["kind"] == kind and row["date"] == date]
            entry = {"kind": kind, "date": date, "samples_per_seed": peers[0]["samples"]}
            for metric in ("rmse", "bias", "explained_variance"):
                values = [row[metric] for row in peers]
                entry.update({f"{metric}_mean": float(np.mean(values)), f"{metric}_sd": float(np.std(values, ddof=1))})
            summary.append(entry)
    with (output / "per-date-summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    export_dates(args.run, output, report)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    for axis, metric, title in zip(axes, ("rmse", "bias", "explained_variance"),
                                   ("MC RMSE (reward units)", "Bias: prediction - MC", "Explained variance"),
                                   strict=True):
        for i, kind in enumerate(COLORS):
            stats = report["groups"][kind]["validation"]
            values = [stats[group][metric]["mean"] for group in GROUPS]
            sd = [stats[group][metric]["sd"] for group in GROUPS]
            axis.errorbar(np.arange(4) + (i - 1) * 0.16, values, yerr=sd, fmt="o-", capsize=3,
                          color=COLORS[kind], label=LABELS[kind], linewidth=1.5)
        if metric != "rmse":
            axis.axhline(0, color="#94a3b8", linewidth=0.8)
        axis.set_xticks(range(4), DATES, rotation=18)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=9)
    figure.suptitle("Frozen joint policy | same 5,904 validation states | final epoch 20\n"
                     "Mean +/- SD across 3 optimizer seeds; not confidence intervals over environments", fontsize=12)
    save(figure, output, "stratified-metrics")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for axis, split in zip(axes, ("train", "validation"), strict=True):
        for kind, color in COLORS.items():
            lines = []
            for seed in report["protocol"]["fit_seeds"]:
                rows = [json.loads(line) for line in
                        (args.run / f"{kind}-{seed}" / "curve.jsonl").read_text(encoding="utf-8").splitlines()]
                lines.append([row[split]["all"]["rmse"] for row in rows])
            values = np.asarray(lines)
            mean, sd = values.mean(0), values.std(0, ddof=1)
            axis.plot(range(len(mean)), mean, color=color, label=LABELS[kind])
            axis.fill_between(range(len(mean)), mean - sd, mean + sd, color=color, alpha=0.12)
        axis.set(title=f"{split.capitalize()} full-terminal MC RMSE", xlabel="Epoch", ylabel="Reward units")
        axis.set_xticks([0, 5, 10, 15, 20])
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    figure.suptitle("Fixed labels and matched optimizer schedules | all epochs shown", fontsize=12)
    save(figure, output, "learning-curves")

    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    months = sorted(report["results"][0]["per_month"])
    for i, kind in enumerate(COLORS):
        peers = [r for r in report["results"] if r["kind"] == kind]
        values = np.asarray([[r["per_month"][month]["rmse"] for month in months] for r in peers])
        axis.errorbar(np.arange(len(months)) + (i - 1) * 0.12, values.mean(0),
                      yerr=values.std(0, ddof=1), fmt="o-", capsize=3, color=COLORS[kind], label=LABELS[kind])
    axis.set_xticks(range(len(months)), months)
    axis.set(ylabel="MC RMSE (reward units)", title="Validation months | mean +/- optimizer-seed SD")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    save(figure, output, "per-month-rmse")

    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["kind", "seed", "split", "date_group", "samples", "rmse", "bias", "explained_variance",
                  "prediction_mean", "return_mean", "optimizer_steps", "loss_samples", "parameters", "seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in report["results"]:
            for split in ("train", "validation"):
                for group, values in result["final"][split].items():
                    writer.writerow({"kind": result["kind"], "seed": result["seed"], "split": split,
                                     "date_group": group, **values,
                                     **{key: result["budget"][key] for key in
                                        ("optimizer_steps", "loss_samples", "parameters")},
                                     "seconds": result["final"]["seconds"]})


if __name__ == "__main__":
    main()
