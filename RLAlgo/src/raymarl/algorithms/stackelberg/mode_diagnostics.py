"""Passive UC mode diagnostics on the actual PPO minibatches; no RNG or updates."""

from itertools import pairwise

import torch
from torch.nn import functional as F

SOC_EDGES = (0.0, 0.1, 0.3, 0.6, 0.9, 1.0)


def flat_surrogate(ratio, advantage, clip):
    """Only the advantage-dependent flat side removes the sample's PPO gradient."""
    return ((advantage > 0) & (ratio > 1 + clip)) | ((advantage < 0) & (ratio < 1 - clip))


def describe(values):
    if not values.numel():
        return None
    values = values.double()
    return {
        "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
        "min": float(values.min()), "max": float(values.max()),
        **{name: float(value) for name, value in zip(
            ("p05", "p50", "p95"), torch.quantile(values, values.new_tensor([0.05, 0.5, 0.95])),
            strict=True,
        )},
    }


def soc_groups(module, batch):
    centered = module.centered_soc(batch["obs"])
    if centered is None:
        return
    soc = (centered.double() + 1) / 2
    for low, high in pairwise(SOC_EDGES):
        yield low, high, (soc >= low) & ((soc < high) if high < 1 else (soc <= high))


@torch.no_grad()
def soc_advantages(module, batch):
    """Count unique rollout rows once, before the Actor epochs repeat them."""
    rows = []
    for low, high, selected in soc_groups(module, batch):
        for mode in range(3):
            mask = selected & (batch["storage_mode"] == mode)
            advantage = batch["advantages"][mask]
            rows.append({
                "soc_low": low, "soc_high": high, "mode": mode, "count": int(mask.sum()),
                "advantage": describe(advantage),
                "raw_advantage": describe(batch["advantages_raw"][mask]) if "advantages_raw" in batch else None,
                "positive_fraction": float((advantage > 0).double().mean()) if mask.any() else None,
            })
    return rows


def mode_gradient(score, inputs, soc, weights):
    """Exact gradient of the linear mode head, bias and optional SOC skip."""
    weighted = weights[:, None] * score / len(score)
    pieces = [(weighted.T @ inputs).flatten(), weighted.sum(0)]
    if soc is not None:
        pieces.append(weighted.T @ soc.double())
    return torch.cat(pieces)


@torch.no_grad()
def mode_statistics(module, batch, output, clip):
    """Counterfactual clipping is diagnostic only; the loss still uses joint PPO."""
    difference = output["logp_parts"].double() - batch["logp_parts"].double()
    ratios = {
        "base": difference[:, 0].exp(), "mode": difference[:, 1].exp(),
        "storage": difference[:, 2].exp(), "continuous": difference[:, [0, 2]].sum(-1).exp(),
        "joint": (output["action_logp"] - batch["action_logp"]).exp(),
    }
    advantage = batch["advantages"].double()
    flat = {key: flat_surrogate(value, advantage, clip) for key, value in ratios.items()}
    probs = output["mode_probs"].double()
    score = F.one_hot(batch["storage_mode"].long(), 3).double() - probs
    inputs = output["mode_inputs"].double()
    soc = module.centered_soc(batch["obs"])
    weights = {
        "joint": -advantage * ratios["joint"] * (~flat["joint"]),
        "unclipped": -advantage * ratios["joint"],
        "separate_mode": -advantage * ratios["mode"] * (~flat["mode"]),
    }
    gradients = {key: mode_gradient(score, inputs, soc, value) for key, value in weights.items()}
    def cosine(left, right):
        denominator = left.norm() * right.norm()
        return float(torch.dot(left, right) / denominator) if denominator > 0 else None
    sample_norm = score.norm(dim=-1) * (inputs.square().sum(-1) + 1 + (
        soc.double().square() if soc is not None else 0
    )).sqrt() * weights["unclipped"].abs()
    total_norm = sample_norm.sum()
    by_soc = [{
        "soc_low": low, "soc_high": high, "count": int(mask.sum()),
        # Contribution to this minibatch's ascent direction; bins sum to the full bias direction.
        "mode_bias_ascent": (-(weights["joint"][:, None] * score)[mask].sum(0) / len(score)).tolist(),
    } for low, high, mask in soc_groups(module, batch)]
    return {
        "samples": len(advantage),
        "ratios": {key: describe(value) for key, value in ratios.items()},
        "range_clip_fraction": {key: float(((value - 1).abs() > clip).double().mean())
                                for key, value in ratios.items()},
        "flat_clip_fraction": {key: float(value.double().mean()) for key, value in flat.items()},
        "mode_blocked_by_joint_fraction": float((flat["joint"] & ~flat["mode"]).double().mean()),
        "removed_mode_gradient_mass_fraction": (
            float(sample_norm[flat["joint"]].sum() / total_norm) if total_norm > 0 else 0.0
        ),
        "policy_mode_gradient_norm": {key: float(value.norm()) for key, value in gradients.items()},
        "joint_vs_separate_cosine": cosine(gradients["joint"], gradients["separate_mode"]),
        "joint_vs_unclipped_cosine": cosine(gradients["joint"], gradients["unclipped"]),
        "mode_bias_ascent": {key: (-(value[:, None] * score).mean(0)).tolist()
                             for key, value in weights.items()},
        "mode_bias_ascent_by_soc": by_soc,
        "joint_logp_parts_max_error": float((output["action_logp"] - output["logp_parts"].sum(-1)).abs().max()),
        "behavior_logp_parts_max_error": float((batch["action_logp"] - batch["logp_parts"].sum(-1)).abs().max()),
    }


@torch.no_grad()
def gradient_norms(module):
    return {name: float(torch.stack([p.grad.double().square().sum() for p in network.parameters()
                                    if p.grad is not None]).sum().sqrt())
            if any(p.grad is not None for p in network.parameters()) else None
            for name, network in module.actor.items()}


def summarize_minibatches(rows):
    """Fractions are weighted by actual sample visits, not just the last minibatch."""
    count = sum(row["samples"] for row in rows)
    def mean(values):
        return sum(row["samples"] * value for row, value in zip(rows, values, strict=True)) / count
    return {
        "minibatches": len(rows), "sample_visits": count,
        "range_clip_fraction": {name: mean([row["range_clip_fraction"][name] for row in rows])
                                for name in ("base", "mode", "storage", "continuous", "joint")},
        "flat_clip_fraction": {name: mean([row["flat_clip_fraction"][name] for row in rows])
                               for name in ("base", "mode", "storage", "continuous", "joint")},
        "mode_blocked_by_joint_fraction": mean([row["mode_blocked_by_joint_fraction"] for row in rows]),
        "removed_mode_gradient_mass_fraction_mean": mean([
            row["removed_mode_gradient_mass_fraction"] for row in rows
        ]),
    } if rows else {"minibatches": 0, "sample_visits": 0}
