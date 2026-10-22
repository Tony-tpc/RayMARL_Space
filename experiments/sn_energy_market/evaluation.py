"""Frozen complete-episode evaluation and true-terminal Critic diagnostics."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.sn_energy_market.training_audit import assert_same_state

ROOT = Path(__file__).resolve().parents[2]
GAMMA = 0.99
VALIDATION_SEEDS = (10042, 10045, 10048, 10051)


def terminal_returns(rewards, terminated, gamma):
    """Original reward units, complete finite episodes, no synthetic terminal."""
    assert terminated[-1]
    result = np.empty_like(rewards, dtype=np.float64)
    carry = 0.0
    for index in reversed(range(len(rewards))):
        carry = float(rewards[index]) + gamma * carry * (not terminated[index])
        result[index] = carry
    return result


def source_hashes(directory):
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / directory).rglob("*.py"))
    }


def cursor(runner):
    state = runner.env.engine.state
    return {
        "seed": runner._scenario_seed,
        "period": state.period_id,
        "timestamp": state.timestamp.isoformat(),
        "terminated": runner._terminated,
        "main_periods": runner.trajectory_periods,
        "soc_kwh": state.ess_energy_kwh,
        "monthly_uc_kwh": sum(c.monthly_uc_energy_kwh for c in state.consumers),
        "tasks": sum(len(c.tasks) for c in state.consumers),
    }


def error_statistics(predictions, targets):
    error = np.asarray(predictions) - np.asarray(targets)
    variance = np.var(targets)
    return {
        "samples": len(error),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "prediction_mean": float(np.mean(predictions)),
        "return_mean": float(np.mean(targets)),
        "explained_variance": float(1 - np.var(error) / variance)
        if variance > 0
        else 0.0,
    }


def stratified_statistics(predictions, targets, periods):
    days = np.asarray(periods) // 48 + 1
    return {
        name: error_statistics(np.asarray(predictions)[mask], np.asarray(targets)[mask])
        for name, mask in {
            "all": days > 0,
            "early_1_10": days <= 10,
            "middle_11_20": (days > 10) & (days <= 20),
            "late_21_end": days > 20,
        }.items()
        if mask.any()
    }


def full_value_validation(algorithm):
    """One frozen trajectory per validation month, every state's true-terminal MC."""
    runner = algorithm.env_runner
    saved = runner.get_state(components="sampling")
    predictions = {key: [] for key in runner.agent_ids}
    targets = {key: [] for key in runner.agent_ids}
    period_ids, matched, episodes = [], [], []
    try:
        for seed in VALIDATION_SEEDS:
            runner.restart(seed)
            rows = {key: [] for key in runner.agent_ids}
            totals, discounted_returns = {}, dict.fromkeys(runner.agent_ids, 0.0)
            period = 0
            while not runner._terminated:
                samples, result = runner._period()
                for key, samples_for_agent in rows.items():
                    samples_for_agent.append(samples[key])
                    discounted_returns[key] += runner.config.gamma**period * result.rewards[key]
                for key, value in result.metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                period += 1
            periods = [row["period_id"] for row in rows["uc"]]
            period_ids.extend(periods)
            episodes.append(
                {"seed": seed, "periods": len(periods), "true_terminal": True,
                 "metrics": totals, "returns": discounted_returns}
            )
            for key, samples in rows.items():
                values = [row["vf_preds"] for row in samples]
                returns = terminal_returns(
                    np.array([row["rewards"] for row in samples]),
                    np.array([row["terminateds"] for row in samples]),
                    runner.config.gamma,
                )
                predictions[key].extend(values)
                targets[key].extend(returns)
                if key == "uc":
                    matched.append(
                        {
                            "seed": seed,
                            "period": 96,
                            "prediction": values[96],
                            "full_terminal_return": float(returns[96]),
                            "sample_error": float(values[96] - returns[96]),
                        }
                    )
            print(
                json.dumps({"validation_month": seed, "periods": len(periods)}),
                flush=True,
            )
    finally:
        runner.set_state(saved)
    return {
        "episodes": episodes,
        "periods": len(period_ids),
        "modules": {
            key: stratified_statistics(predictions[key], targets[key], period_ids)
            for key in predictions
        },
        "matched_period_96": matched,
        "matched_period_96_rmse": float(
            np.sqrt(np.mean([r["sample_error"] ** 2 for r in matched]))
        ),
        "scope": "Single frozen stochastic MC trajectory per month; correlated samples, not exact expectations.",
    }


def retained_optimizer_steps(learner):
    return {
        name: sorted(
            {
                int(state["step"].item())
                for state in optimizer.state.values()
                if "step" in state
            }
        )
        for name, optimizer in learner._named_optimizers.items()
    }


def frozen_evaluation(algorithm):
    before = copy.deepcopy(algorithm.learner_group.get_state())
    result = {
        "performance": {
            "sampled": algorithm.evaluate_pair(VALIDATION_SEEDS),
            "mean": algorithm.evaluate_pair(VALIDATION_SEEDS, deterministic=True),
        },
        "critic": full_value_validation(algorithm),
    }
    # Includes every optimizer, not only policy weights.
    assert_same_state(before, algorithm.learner_group.get_state())
    return result
