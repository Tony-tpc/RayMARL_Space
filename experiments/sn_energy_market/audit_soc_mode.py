"""Read saved UC weights and measure SOC response, without sampling or learning."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RLAlgo/src"))
from raymarl.algorithms.stackelberg.rl_module import StackelbergRLModule


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


def main(output, previous):
    output, previous = Path(output), Path(previous)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    source = Path(protocol["fixed_state_probe_source"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["fixed_state_probe_sha256"]
    paths, results = [source, output / "protocol.json"], {}
    final_round = json.loads((output / "report.json").read_text(encoding="utf-8"))["outer_rounds"]
    system = protocol["physical_config"]["system"]
    torch.set_num_threads(1)
    with np.load(source) as data, torch.random.fork_rng(devices=[]):
        obs, bounds = torch.tensor(data["obs"]), torch.tensor(data["bounds"])
        base = torch.tensor(data["controls"][:, [0, 1, 5]])
        for label, directory, number in (("previous_initial", previous, 0), ("previous", previous, 10),
                                         ("initial", output, 0), ("round3", output, 3), ("current", output, final_round)):
            report_path = directory / "report.json"
            contract = json.loads(report_path.read_text(encoding="utf-8"))["learning_contract"]
            transform = contract["input_transform"]
            metadata = transform.get("uc_actor_soc") if isinstance(transform, dict) else None
            config = {**contract["model"], "role": "uc"}
            if metadata is not None:
                config["soc_input"] = metadata
            module = RLModuleSpec(
                module_class=StackelbergRLModule,
                observation_space=gym.spaces.Box(-np.inf, np.inf, (obs.shape[1],), np.float32),
                action_space=gym.spaces.Box(-np.inf, np.inf, (6,), np.float64), model_config=config,
            ).build()
            state_path = directory / f"data/uc-state-{number:04d}.npz"
            with np.load(state_path) as state:
                module.load_state_dict({key: torch.tensor(state[key]) for key in state.files}, strict=True)
            soc_input = metadata or {"index": 2, "min": system["ess_energy_min_kwh"],
                                     "max": system["ess_energy_capacity_kwh"]}
            actual, result = probe(module, obs, bounds, base, soc_input)
            fixed_path = directory / f"data/fixed-state-probe-{number:04d}.npz"
            with np.load(fixed_path) as fixed:
                np.testing.assert_allclose(actual, fixed["probs"], atol=1e-7, rtol=0)
                result["fixed_probability_replay_max_error"] = float(np.abs(actual - fixed["probs"]).max())
            results[label] = result
            paths.extend((report_path, state_path, fixed_path))
    target = output / "soc-mode-audit.json"
    target.write_text(json.dumps({"policies": results, "outer_rounds": final_round, "no_sampling_or_learning": True,
        "scope": "interior SOC partial response; fixed other observations, base controls and feasibility",
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},
        ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"soc_mode_audit": str(target)}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
