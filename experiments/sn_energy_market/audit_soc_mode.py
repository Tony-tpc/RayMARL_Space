"""Frozen SOC response diagnostic, independent of historical experiments."""
import numpy as np
import torch


@torch.no_grad()
def probe(module, obs, bounds, base, soc_input):
    """Vary only interior SOC; hold the original feasible mask and base action fixed.

    This is an Actor partial-response test, not a rollout of counterfactual physics.
    Boundary-forced mode switches are excluded from the learned-switch statistic.
    """
    distribution, mask = module._uc_mode(module._features(obs), base, bounds, module.centered_soc(obs))
    legal = mask.all(-1)
    if not legal.any():
        raise ValueError("SOC probe requires states with all three modes feasible")
    fractions = np.linspace(.05, .95, 19)
    probabilities = []
    for fraction in fractions:
        x = obs.clone()
        x[:, soc_input["index"]] = soc_input["min"] + fraction * (soc_input["max"] - soc_input["min"])
        conditional, changed_mask = module._uc_mode(module._features(x), base, bounds, module.centered_soc(x))
        assert torch.equal(mask, changed_mask)
        probabilities.append(conditional._dist.probs[legal].cpu().numpy())
    p = np.stack(probabilities)
    argmax = p.argmax(-1)
    return distribution._dist.probs.cpu().numpy(), {
        "states": int(legal.sum()), "soc_fractions": fractions.tolist(),
        "probability_mean": p.mean(1).tolist(), "probability_std": p.std(1).tolist(),
        "argmax_counts": [np.bincount(row, minlength=3).tolist() for row in argmax],
        "switch_fraction": float(np.any(argmax != argmax[0], axis=0).mean()),
        "low_high_transitions": np.bincount(argmax[0] * 3 + argmax[-1], minlength=9).reshape(3, 3).tolist(),
        "mean_probability_range": np.ptp(p, axis=0).mean(0).tolist(),
        "soc_skip_weights": module.actor["mode_soc"]["weight"].flatten().cpu().tolist()
            if "mode_soc" in module.actor else None,
    }


