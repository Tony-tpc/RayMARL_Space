"""Check native game coupling using paired real environment trajectories."""
import json
from pathlib import Path
import numpy as np
from city_proxy import CityLearnProxy, ROOT


def action(env, battery):
    actions = {}
    for i, a in enumerate(env.agents):
        values = np.zeros(len(env.continuous[i]), np.float32)
        names = [env.action_names[i][j] for j in env.continuous[i]]
        values[names.index("electrical_storage")] = battery.get(a, 0)
        actions[a] = {"continuous": values, "binary": np.zeros(len(env.binary[i]), np.int8)} if (
            env.binary[i]) else values
    return actions


def main():
    config = {"track": "hybrid", "split": "probe", "episode_steps": 48, "seed": 0}
    a, b = CityLearnProxy(config), CityLearnProxy(config)
    try:
        o1, _ = a.reset(seed=0)
        o2, _ = b.reset(seed=0)
        assert all(np.array_equal(o1[k]["current"], o2[k]["current"]) for k in a.agents)
        for _ in range(3):
            prefix = {"Building_1": .7}
            a.step(action(a, prefix));b.step(action(b, prefix))
        _, r1, _, _, _ = a.step(action(a, {"Building_1": -1, "Building_2": 1}))
        _, r2, _, _, _ = b.step(action(b, {"Building_1": -1, "Building_2": 0}))
        delta = {k: r2[k]-r1[k] for k in r1}
        report = {"same_initial_local_observations": True,
            "fixed_own_action": "Building_1 electrical_storage=-1",
            "perturbed_other_action": "Building_2 electrical_storage 1 -> 0",
            "reward_before": r1, "reward_after": r2, "reward_delta": delta,
            "coupling_passed": abs(delta["Building_1"]) > 1e-8,
            "conflict_passed": delta["Building_1"]*delta["Building_2"] < 0}
        destination = ROOT/"outputs"/"preflight.json"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not report["coupling_passed"] or not report["conflict_passed"]:
            raise RuntimeError("Native non-cooperative game qualification failed")
    finally:
        a.close();b.close()


if __name__ == "__main__":
    main()
