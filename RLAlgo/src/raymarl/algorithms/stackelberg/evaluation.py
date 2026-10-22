"""Matched evaluation, candidate selection and joint stopping diagnostics."""

import numpy as np


def paired_gain(candidate, incumbent, agent="uc", z=1.96):
    """Approximate normal interval over independent, matched scenario seeds."""
    if [row["seed"] for row in candidate] != [row["seed"] for row in incumbent]:
        raise ValueError("Candidate evaluation requires the same ordered scenario seeds")
    if any(not row["score_complete"] for row in (*candidate, *incumbent)):
        raise ValueError("Candidate comparison requires complete terminal returns; windows are diagnostic only")
    gains = np.asarray(
        [a["returns"][agent] - b["returns"][agent] for a, b in zip(candidate, incumbent, strict=True)],
        dtype=np.float64,
    )
    mean = float(gains.mean())
    se = float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else None
    return {
        "mean": mean,
        "standard_error": se,
        "lower": mean - z * se if se is not None else None,
        "upper": mean + z * se if se is not None else None,
        "seeds": len(gains),
    }


def select_candidate(evaluations, *, z=1.96, min_gain=0.0):
    comparisons = [paired_gain(rows, evaluations[0], z=z) for rows in evaluations]
    eligible = [
        i
        for i in range(1, len(evaluations))
        if comparisons[i]["lower"] is not None and comparisons[i]["lower"] > min_gain
    ]
    selected = max(eligible, key=lambda i: comparisons[i]["mean"]) if eligible else 0
    return selected, comparisons


def summarize_evaluation(rows):
    agents = tuple(rows[0]["returns"])
    return {
        "scenarios": len(rows),
        "physical_periods": sum(row["physical_periods"] for row in rows),
        "returns": {key: float(np.mean([row["returns"][key] for row in rows])) for key in agents},
        "metrics": {key: float(np.mean([row["metrics"][key] for row in rows])) for key in rows[0]["metrics"]},
        "bootstrap_used": any(row["bootstrap_used"] for row in rows),
    }


def actor_parameter_change(before_weights, after_weights):
    """Per-agent RMS over all actor scalars, then the maximum across agents.

    Value-network weights do not enter the policy-change measure. Each agent's
    RMS weights every scalar equally, rather than giving small and large tensors
    the same weight. Inputs are native LearnerGroup module weight dictionaries.
    """
    if before_weights.keys() != after_weights.keys():
        raise ValueError("Policy-change comparison requires the same agent layout")
    changes = {}
    for key, before in before_weights.items():
        after = after_weights[key]
        names = {name for name in before if name.startswith("actor.")}
        if not names or names != {name for name in after if name.startswith("actor.")}:
            raise ValueError("Policy-change comparison requires matching actor parameters")
        total, count = 0.0, 0
        for name in sorted(names):
            a, b = before[name], after[name]
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            if hasattr(b, "detach"):
                b = b.detach().cpu().numpy()
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if a.shape != b.shape:
                raise ValueError("Policy-change comparison requires matching parameter shapes")
            total += float(np.square(b - a).sum())
            count += a.size
        changes[key] = float(np.sqrt(total / count))
    return {"per_agent": changes, "maximum_rms": max(changes.values())}


def stopping_status(consumer_deviations, uc_comparisons, policy_change, tolerances):
    """Check the document's four joint criteria for the measured search budget.

    ``uc_comparisons`` includes the incumbent at index zero. Both leader and
    follower searches need paired confidence intervals; skipped searches or a
    single evaluation seed cannot certify the stopping rule.
    """
    followers = list(consumer_deviations["per_consumer"].values())
    comparisons = followers + list(uc_comparisons)
    searched = consumer_deviations["searched"] and bool(followers) and len(uc_comparisons) > 1
    intervals_available = bool(comparisons) and all(
        all(row.get(key) is not None and np.isfinite(row[key]) for key in ("mean", "lower", "upper"))
        and row["lower"] <= row["upper"]
        for row in comparisons
    )
    measurements = {
        "consumer_gain": max((row["found_gain"] for row in followers), default=None),
        "uc_gain": max((max(0.0, row["mean"]) for row in uc_comparisons), default=None),
        "policy_change": policy_change["maximum_rms"],
        "confidence_half_width": max((row["upper"] - row["lower"]) / 2 for row in comparisons)
        if intervals_available
        else None,
    }
    criteria = {"deviations_searched": bool(searched), "intervals_available": bool(intervals_available)}
    criteria.update(
        {
            key: bool(value is not None and np.isfinite(value) and value <= tolerances[key])
            for key, value in measurements.items()
        }
    )
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "measurements": measurements,
        "tolerances": dict(tolerances),
    }
