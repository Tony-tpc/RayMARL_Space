"""Plot measured full-batch losses by collection ID, including discarded branches."""

import argparse
import json
from pathlib import Path


def load_batches(path):
    batches = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            index = row["batch_id"]
            if row["event"] == "batch":
                if index in batches:
                    raise ValueError(f"Duplicate batch ID: {index}")
                batches[index] = row
            elif row["event"] == "branch_resolution":
                if index not in batches:
                    raise ValueError(f"Resolution without a measured batch: {index}")
                batches[index].update({key: value for key, value in row.items() if key != "event"})
            else:
                raise ValueError(f"Unknown batch event: {row['event']}")
    if not batches:
        raise ValueError("No measured batch losses; historical round means cannot supply batch losses")
    return [batches[index] for index in sorted(batches)]


def plot_batches(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = [row["batch_id"] for row in rows]
    paths = []
    for key in rows[0]["modules"]:
        figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        for axis, prefix, role, label in zip(
            axes, ("critic_loss", "critic_rmse", "policy_loss"), ("critic", "critic", "actor"),
            ("Critic loss (scaled half MSE)", "Critic RMSE (reward units)", "Actor clipped policy loss"),
            strict=True,
        ):
            for suffix, style in (("before", "--"), ("after", "-")):
                values = [row["modules"][key][f"{prefix}_{suffix}"] for row in rows]
                axis.plot(ids, [float("nan") if v is None else v for v in values],
                          style, marker=".", label=suffix)
            discarded, pending = [], []
            for row in rows:
                value = row["modules"][key][f"{prefix}_after"]
                if value is None:
                    continue
                if row["branch_status"] == "pending":
                    pending.append((row["batch_id"], value))
                elif not row["retained_updates"][key][role]:
                    discarded.append((row["batch_id"], value))
            for points, marker, text in ((discarded, "x", "rolled back"), (pending, "s", "pending")):
                if points:
                    axis.scatter(*zip(*points, strict=True), marker=marker, color="crimson", label=text)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            axis.legend(loc="best", ncol=4)
        axes[-1].set_xlabel("Collected batch ID (includes candidate branches; frozen actors are gaps)")
        axes[-1].set_xticks(ids)
        axes[-1].set_xlim(ids[0] - 0.5, ids[-1] + 0.5)
        figure.suptitle(f"{key}: fixed targets within each batch; targets differ between batches")
        # Module IDs can contain path separators; keep every output inside the directory.
        safe_key = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)
        for extension in ("png", "svg"):
            path = output / f"batch-loss-{safe_key}.{extension}"
            figure.savefig(path, dpi=160)
            paths.append(path)
        plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Native run's batch_losses.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_batches(args.log)
    for path in plot_batches(rows, args.output):
        print(path)


if __name__ == "__main__":
    main()
