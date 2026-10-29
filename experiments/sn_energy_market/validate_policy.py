"""Frozen complete-month validation; no labels enter training."""
import json
import random
from collections import defaultdict

import numpy as np
import torch

from experiments.sn_energy_market.evaluation import error_statistics, stratified_statistics, terminal_returns
from experiments.sn_energy_market.run_io import write_csv, write_json
from experiments.sn_energy_market.runtime import EVALUATION_SEEDS, MONTH_SEEDS, digest_tree
from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner, random_state, restore_random_state


@torch.inference_mode()
def complete_mc(algorithm, target, progress=None):
    """Separate evaluation runner: MC labels never reach the learner/advantages."""
    target.mkdir(parents=True, exist_ok=True)
    before = digest_tree(algorithm.get_state())
    outside = random_state()
    config = algorithm.config.copy(copy_frozen=False)
    config.execution_workers = 0
    runner = StackelbergEnvRunner(config=config)
    runner.module.set_state(algorithm.learner_group.get_weights())
    rows, episodes = [], []
    consumer_values, consumer_targets, consumer_periods = [], [], []
    per_consumer = defaultdict(lambda: {"values": [], "targets": [], "periods": [], "months": []})
    try:
        for scenario in MONTH_SEEDS:
            for action_seed in EVALUATION_SEEDS:
                runner.restart(scenario)
                seed = action_seed * 1_000_003 + scenario
                torch.manual_seed(seed)
                np.random.seed(seed % 2**32)
                random.seed(seed)
                runner._action_rng = random_state()
                uc_rows, consumers, totals = (
                    [],
                    {k: [] for k in runner.env.consumer_ids},
                    defaultdict(float),
                )
                discounted = dict.fromkeys(runner.agent_ids, 0.0)
                t = 0
                uc_inputs = []
                while not runner._terminated:
                    stamp = runner.env.engine.state.timestamp.isoformat()
                    samples, result = runner._period(with_values=False)
                    uc = samples["uc"]
                    uc_inputs.append(uc["critic_obs"])
                    uc_rows.append(
                        {
                            "scenario_seed": scenario,
                            "action_seed": action_seed,
                            "timestamp": stamp,
                            "period": t,
                            "soc_kwh": float(uc["obs"][2]),
                            "mode": int(uc["storage_mode"]),
                            "prediction": 0.0,
                            "reward": float(result.rewards["uc"]),
                            "grid_charge_kwh": float(result.metrics["grid_charge_kwh"]),
                            "der_charge_kwh": float(result.metrics["der_charge_kwh"]),
                            "ess_discharge_kwh": float(result.metrics["ess_discharge_kwh"]),
                        }
                    )
                    for key, value in result.rewards.items():
                        discounted[key] += algorithm.config.gamma**t * value
                    for key, observations in consumers.items():
                        observations.append(
                            (samples[key]["critic_obs"], result.rewards[key])
                        )
                    for key, value in result.metrics.items():
                        totals[key] += value
                    t += 1
                flags = np.zeros(t, dtype=bool)
                flags[-1] = True
                for row, value in zip(uc_rows, runner._values("uc", uc_inputs), strict=True):
                    row["prediction"] = float(value)
                mc = terminal_returns(
                    np.asarray([r["reward"] for r in uc_rows]),
                    flags,
                    algorithm.config.gamma,
                )
                # Independent direct-sum checks on early/mid/last labels.
                rewards = np.asarray([r["reward"] for r in uc_rows])
                for index in (0, t // 2, t - 1):
                    np.testing.assert_allclose(
                        mc[index],
                        np.dot(
                            algorithm.config.gamma ** np.arange(t - index),
                            rewards[index:],
                        ),
                        atol=1e-9,
                    )
                for row, value in zip(uc_rows, mc, strict=True):
                    row["mc"] = float(value)
                rows.extend(uc_rows)
                for key, data in consumers.items():
                    values = runner._values(key, [item[0] for item in data])
                    rewards = np.asarray([item[1] for item in data])
                    targets_for_agent = terminal_returns(rewards, flags, algorithm.config.gamma)
                    consumer_values.extend(values)
                    consumer_targets.extend(targets_for_agent)
                    consumer_periods.extend(range(t))
                    per_consumer[key]["values"].extend(values)
                    per_consumer[key]["targets"].extend(targets_for_agent)
                    per_consumer[key]["periods"].extend(range(t))
                    per_consumer[key]["months"].extend([uc_rows[0]["timestamp"][:7]] * t)
                episodes.append(
                    {
                        "scenario_seed": scenario,
                        "action_seed": action_seed,
                        "periods": t,
                        "true_terminal": True,
                        "returns": discounted,
                        "metrics": dict(totals),
                    }
                )
                print(
                    json.dumps(
                        {
                            "evaluation": str(target),
                            "month": scenario,
                            "action_seed": action_seed,
                            "periods": t,
                        }
                    ),
                    flush=True,
                )
                if progress:
                    progress(scenario=scenario, action_seed=action_seed)
    finally:
        runner.stop()
        restore_random_state(outside)
    assert digest_tree(algorithm.get_state()) == before, (
        "Evaluation changed training state"
    )
    values, targets, periods = [
        np.asarray([r[k] for r in rows]) for k in ("prediction", "mc", "period")
    ]
    by_date, by_month = [], []
    for field, output in (("date", by_date), ("month", by_month)):
        keys = [r["timestamp"][: 10 if field == "date" else 7] for r in rows]
        for name in sorted(set(keys)):
            mask = np.asarray([k == name for k in keys])
            output.append(
                {field: name, **error_statistics(values[mask], targets[mask])}
            )
    summary = {
        "periods": len(rows),
        "episodes": episodes,
        "uc": stratified_statistics(values, targets, periods),
        "consumer_pooled": stratified_statistics(
            consumer_values, consumer_targets, consumer_periods
        ),
        "macro_month_rmse": float(np.mean([r["rmse"] for r in by_month])),
        "worst_month_rmse": float(max(r["rmse"] for r in by_month)),
        "by_month": by_month,
        "consumers": {key: {
            **stratified_statistics(data["values"], data["targets"], data["periods"]),
            "by_month": [{"month": month, **error_statistics(
                np.asarray(data["values"])[np.asarray(data["months"]) == month],
                np.asarray(data["targets"])[np.asarray(data["months"]) == month])}
                for month in sorted(set(data["months"]))],
        } for key, data in per_consumer.items()},
        "training_state_unchanged": True,
        "mc_training_labels": 0,
        "bootstrap_used": False,
    }
    write_csv(target / "uc-trajectories.csv", rows)
    write_csv(target / "by-date.csv", by_date)
    write_csv(target / "by-month.csv", by_month)
    write_json(target / "summary.json", summary)
    return summary


