"""Frozen validation to distinguish Critic error from a truncated return tail."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "RLAlgo/src"),
    str(ROOT / "environments/sn_energy_market/src"),
]

from experiments.sn_energy_market.calibrated_pilot import GAMMA, OUTPUT
from experiments.sn_energy_market.pilot import parameter_changes
from experiments.sn_energy_market.train import write_json


def main():
    import torch
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
    from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner

    torch.set_num_threads(1)
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    checkpoint = OUTPUT / "checkpoint"
    runner = StackelbergEnvRunner.from_checkpoint(checkpoint / "env_runner")
    runner.module = MultiRLModule.from_checkpoint(
        checkpoint / "learner_group/learner/rl_module"
    )
    assert set(runner.config.scenario_factory.dataset.splits) == {"train", "validation"}
    weights = copy.deepcopy(runner.module.get_state())
    rows = []
    periods = 0
    for reference in report["value_calibration_final"]["rows"]:
        seed = reference["seed"]
        assert seed in runner.config.validation_seeds
        runner.restart(seed)
        for _ in range(96):
            runner._period()
            periods += 1
        prediction = runner._value("uc", runner._observation, runner._observation)
        np.testing.assert_allclose(prediction, reference["prediction"], rtol=0, atol=1e-10)
        rewards = []
        while not runner._terminated:
            _, transition = runner._period()
            rewards.append(transition.rewards["uc"])
            periods += 1
        discounted = np.asarray(rewards) * GAMMA ** np.arange(len(rewards))
        prefix = float(discounted[:96].sum())
        np.testing.assert_allclose(
            prefix, reference["following_96_step_return"], rtol=0, atol=1e-10
        )
        full = float(discounted.sum())
        row = {
            "seed": seed,
            "prediction": prediction,
            "following_periods_to_true_terminal": len(rewards),
            "first_96_return": prefix,
            "discounted_tail": float(discounted[96:].sum()),
            "full_terminal_return": full,
            "sample_error": prediction - full,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    changes = parameter_changes(weights, runner.module.get_state())
    assert all(value == 0 for pair in changes.values() for value in pair.values())
    write_json(
        OUTPUT / "value_tail.json",
        {
            "status": "passed",
            "test_loaded": False,
            "training_updates": 0,
            "validation_periods": periods,
            "weights_unchanged": True,
            "original_96_step_values_reproduced": True,
            "rows": rows,
            "full_terminal_sample_rmse": float(
                np.sqrt(np.mean([row["sample_error"] ** 2 for row in rows]))
            ),
            "scope": "One frozen stochastic trajectory per validation month; MC samples, not exact expected values.",
        },
    )
    runner.stop()


if __name__ == "__main__":
    main()
