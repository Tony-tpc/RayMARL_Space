"""Recompute saved probe evidence without sampling or updating any model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RLAlgo/src"))

from raymarl.algorithms.stackelberg.critic_probe import CriticProbe


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(prediction, target):
    error = np.asarray(prediction, dtype=np.float64) - target
    return {"rmse": float(np.sqrt(np.dot(error, error) / len(error))),
            "bias": float(error.sum() / len(error)),
            "explained_variance": float(1 - np.var(error) / np.var(target))}


def masks(periods):
    return {"all": periods >= 0, "early_1_10": periods < 480,
            "middle_11_20": (periods >= 480) & (periods < 960),
            "late_21_end": periods >= 960}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    protocol = report["protocol"]
    manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
    episodes, label_summary = [], []
    maximum_label_difference = 0.0
    for entry in manifest:
        path = run / "data" / entry["file"]
        assert digest(path) == entry["sha256"]
        with np.load(path, allow_pickle=False) as data:
            row = {name: data[name] for name in data.files}
        row["metadata"] = entry
        n = len(row["rewards"])
        assert row["terminated"].sum() == 1 and row["terminated"][-1]
        np.testing.assert_array_equal(row["periods"], np.arange(n))
        # Direct discounted sums at separated positions, independent of the
        # backwards recurrence used to create labels; recursion checks all rows.
        for index in (0, n // 2, n - 1):
            direct = np.dot(row["rewards"][index:], protocol["gamma"] ** np.arange(n - index))
            maximum_label_difference = max(maximum_label_difference, abs(direct - row["targets"][index]))
            np.testing.assert_allclose(direct, row["targets"][index], rtol=0, atol=1e-10)
        np.testing.assert_allclose(row["targets"], row["rewards"] + protocol["gamma"] *
                                   np.r_[row["targets"][1:], 0.0], rtol=0, atol=1e-10)
        assert np.isfinite(row["targets"]).all()
        label_summary.append({"month": entry["month"], "split": entry["split"], "samples": n,
                              "mean": float(row["targets"].mean()), "std": float(row["targets"].std()),
                              "minimum": float(row["targets"].min()),
                              "minimum_timestamp": str(row["timestamps"][row["targets"].argmin()]),
                              "negative_100_samples": int((row["targets"] < -100).sum())})
        episodes.append(row)
    by_split = {split: [e for e in episodes if e["metadata"]["split"] == split]
                for split in ("train", "validation")}
    model_audit, maximum_metric_difference = [], 0.0
    for result in report["results"]:
        kind, seed = result["kind"], result["seed"]
        directory = run / f"{kind}-{seed}"
        checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu")
        model = CriticProbe(kind, 68, seed=seed).eval()
        model.load_state_dict(checkpoint["model"])
        assert sum(p.numel() for p in model.parameters()) == result["budget"]["parameters"]
        adam_steps = {int(state["step"].item()) for state in checkpoint["optimizer"]["state"].values()}
        assert adam_steps == {1380}
        assert result["budget"]["loss_samples"] == 20 * 17520
        assert result["final"]["epoch"] == 20
        train_sse, per_month, sum_sse = {}, {}, 0.0
        for split, selected in by_split.items():
            predictions = []
            with np.load(directory / "validation_predictions.npz", allow_pickle=False) as saved:
                for row in selected:
                    prediction = model.predict_episode(torch.as_tensor(row["obs"], dtype=torch.float32),
                                                       torch.as_tensor(row["calendar"], dtype=torch.float32)).numpy()
                    predictions.append(prediction)
                    if split == "validation":
                        np.testing.assert_array_equal(prediction, saved[str(row["metadata"]["seed"])])
                    else:
                        error = prediction.astype(np.float64) - row["targets"]
                        train_sse[row["metadata"]["month"]] = float(np.dot(error, error))
                        sum_sse += float(np.dot(error, error))
                    per_month[row["metadata"]["month"]] = {"split": split,
                                                           **metrics(prediction, row["targets"])}
            prediction = np.concatenate(predictions)
            target = np.concatenate([row["targets"] for row in selected])
            periods = np.concatenate([row["periods"] for row in selected])
            for group, mask in masks(periods).items():
                recomputed = metrics(prediction[mask], target[mask])
                for key, value in recomputed.items():
                    difference = abs(value - result["final"][split][group][key])
                    maximum_metric_difference = max(maximum_metric_difference, difference)
                    assert difference < 1e-10, (kind, seed, split, group, key, difference)
        model_audit.append({"kind": kind, "seed": seed, "adam_steps": sorted(adam_steps),
                            "checkpoint_sha256": digest(directory / "checkpoint.pt"),
                            "predictions_sha256": digest(directory / "validation_predictions.npz"),
                            "per_month": per_month,
                            "training_sse_fraction": {month: sse / sum_sse for month, sse in train_sse.items()}})
    for seed in protocol["fit_seeds"]:
        schedules = {r["budget"]["schedule_sha256"] for r in report["results"] if r["seed"] == seed}
        assert len(schedules) == 1
    for name, expected in protocol["source_sha256"].items():
        assert digest(ROOT / name) == expected, name
    assert digest(ROOT / protocol["source"] / "actors/actors.pt") == protocol["actor_file_sha256"]
    audit = {"passed": True, "episodes": len(episodes), "models": len(model_audit),
             "report_sha256": digest(run / "report.json"),
             "maximum_direct_MC_difference": maximum_label_difference,
             "maximum_metric_difference": maximum_metric_difference,
             "saved_prediction_replay_exact": True, "source_and_actors_unchanged": True,
             "labels": label_summary, "model_audit": model_audit}
    (run / "postrun_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key not in ("labels", "model_audit")}))


if __name__ == "__main__":
    main()
