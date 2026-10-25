"""Read-only complete-month MC and common-state mode interventions."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from raymarl.algorithms.stackelberg.runner import (
    StackelbergEnvRunner,
    random_state,
    restore_random_state,
)

from experiments.sn_energy_market.evaluation import (
    error_statistics,
    stratified_statistics,
    terminal_returns,
)
from experiments.sn_energy_market.sampling_gate import ROOT
from experiments.sn_energy_market.sampling_gate.io import (
    read_json,
    write_csv,
    write_json,
)
from experiments.sn_energy_market.sampling_gate.runtime import (
    EVALUATION_SEEDS,
    MONTH_SEEDS,
    digest_tree,
)

MODE_SOURCE = ROOT / "outputs/diagnostics/20261024-frozen72-order-mode/mode"
MODE_REPLICATIONS = (824001, 824002, 824003, 824004)


@torch.inference_mode()
def complete_mc(algorithm, target, progress=None):
    """Separate evaluation runner: MC labels never reach the learner/advantages."""
    target.mkdir(parents=True, exist_ok=True)
    before = digest_tree(algorithm.get_state())
    outside = random_state()
    runner = StackelbergEnvRunner(config=algorithm.config)
    runner.module.set_state(algorithm.learner_group.get_weights())
    rows, episodes = [], []
    consumer_values, consumer_targets, consumer_periods = [], [], []
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
                while not runner._terminated:
                    stamp = runner.env.engine.state.timestamp.isoformat()
                    samples, result = runner._period()
                    uc = samples["uc"]
                    uc_rows.append(
                        {
                            "scenario_seed": scenario,
                            "action_seed": action_seed,
                            "timestamp": stamp,
                            "period": t,
                            "soc_kwh": float(uc["obs"][2]),
                            "mode": int(uc["storage_mode"]),
                            "prediction": float(uc["vf_preds"]),
                            "reward": float(result.rewards["uc"]),
                        }
                    )
                    for key, value in result.rewards.items():
                        discounted[key] += algorithm.config.gamma**t * value
                    for key, observations in consumers.items():
                        observations.append(
                            (samples[key]["vf_preds"], result.rewards[key])
                        )
                    for key, value in result.metrics.items():
                        totals[key] += value
                    t += 1
                flags = np.zeros(t, dtype=bool)
                flags[-1] = True
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
                for data in consumers.values():
                    values, rewards = np.asarray(data).T
                    consumer_values.extend(values)
                    consumer_targets.extend(
                        terminal_returns(rewards, flags, algorithm.config.gamma)
                    )
                    consumer_periods.extend(range(t))
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
        "training_state_unchanged": True,
        "mc_training_labels": 0,
        "bootstrap_used": False,
    }
    write_csv(target / "uc-trajectories.csv", rows)
    write_csv(target / "by-date.csv", by_date)
    write_csv(target / "by-month.csv", by_month)
    write_json(target / "summary.json", summary)
    return summary


def seed_draw(rep_seed, scenario_seed, period, agent_index, state_period):
    value = (
        rep_seed * 1_000_000_007
        + scenario_seed * 1_000_003
        + (state_period + 1) * 1_000_033
        + period * 1009
        + agent_index * 37
    )
    torch.random.default_generator.manual_seed(value % (2**63 - 1))


def tensors(obs, bounds):
    return torch.as_tensor(obs, dtype=torch.float32)[None], torch.as_tensor(
        bounds, dtype=torch.float64
    )[None]


@torch.inference_mode()
def mode_controls(module, env, observation):
    obs, bounds = tensors(observation, env.uc_bounds())
    features = module._features(obs)
    base = module._uc_base(features, bounds).loc
    distribution, mask = module._uc_mode(
        features, base, bounds, module.centered_soc(obs)
    )
    controls = []
    for mode in range(3):
        m = torch.tensor([mode], dtype=torch.long)
        storage = module._gaussian(
            module._uc_storage_parameters(features, base, m),
            torch.ones_like(base, dtype=torch.bool),
        ).loc
        raw = torch.stack(
            (
                base[..., 0],
                base[..., 1],
                storage[..., 0],
                storage[..., 1],
                storage[..., 2],
                base[..., 2],
            ),
            -1,
        )
        intervals = module._uc_intervals(bounds, raw, m)
        raw = torch.where(intervals[..., 1] > 0, raw, 0.0)
        controls.append(torch.cat((m[..., None].to(raw.dtype), raw), -1)[0].numpy())
    return controls, distribution._dist.probs[0].numpy(), mask[0].numpy()


@torch.inference_mode()
def common_state_modes(algorithm, target, progress=None):
    """Reuse physical snapshots only; never load old policy or optimizer weights."""
    target.mkdir(parents=True, exist_ok=True)
    before = digest_tree(algorithm.get_state())
    outside = random_state()
    selected = read_json(MODE_SOURCE / "selected-states.json")["states"]
    manifest = {
        r["state_id"]: r for r in read_json(MODE_SOURCE / "snapshot-manifest.json")
    }
    modules, factory = algorithm.env_runner.module, algorithm.config.scenario_factory
    rows, states = [], []
    try:
        for state in sorted(selected, key=lambda x: x["state_id"]):
            item = manifest[state["state_id"]]
            raw = (
                MODE_SOURCE / "snapshots" / f"state-{state['state_id']:02d}.pkl"
            ).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == item["sha256"]
            snapshot = pickle.loads(raw)
            scenario, start = state["scenario_seed"], state["period"]
            env = factory(scenario)
            env.set_state(snapshot)
            observation = env.engine.uc_observation()
            controls, probs, mask = mode_controls(modules["uc"], env, observation)
            assert mask.all()
            obs = torch.as_tensor(observation, dtype=torch.float32)[None]
            prediction = float(modules["uc"].compute_values({"obs": obs}).item())
            states.append(
                {
                    **state,
                    "prediction": prediction,
                    "actor_argmax": int(probs.argmax()),
                    **{
                        f"prob_{name}": float(probs[i])
                        for i, name in enumerate(("idle", "charge", "discharge"))
                    },
                }
            )
            for rep in MODE_REPLICATIONS:
                for mode in (-1, 0, 1, 2):
                    env.set_state(snapshot)
                    observation = env.engine.uc_observation()
                    gain, totals, offset = 0.0, defaultdict(float), 0
                    while True:
                        absolute = env.engine.state.period_id
                        if offset == 0 and mode >= 0:
                            action = controls[mode]
                        else:
                            seed_draw(rep, scenario, absolute, 0, start)
                            action = (
                                modules["uc"]
                                .draw_action(
                                    *tensors(observation, env.uc_bounds()), False
                                )[0]
                                .numpy()
                            )
                        decision = env.publish(action)
                        actions = {}
                        for index, key in enumerate(env.consumer_ids, 1):
                            seed_draw(rep, scenario, absolute, index, start)
                            actions[key] = (
                                modules[key]
                                .draw_action(
                                    *tensors(
                                        decision.consumer_observations[key],
                                        decision.consumer_bounds[key],
                                    ),
                                    False,
                                )[0]
                                .numpy()
                            )
                        result = env.step(actions)
                        gain += algorithm.config.gamma**offset * result.rewards["uc"]
                        for name in (
                            "uc_cash",
                            "native_lost_kwh",
                            "consumer_cost",
                            "terminal_energy_value",
                        ):
                            totals[name] += result.metrics[name]
                        offset += 1
                        if result.terminated:
                            break
                        observation = result.next_uc_observation
                    rows.append(
                        {
                            "state_id": state["state_id"],
                            "replication": rep,
                            "mode": mode,
                            "return": gain,
                            "periods": offset,
                            **totals,
                        }
                    )
            write_csv(target / "rollouts.csv", rows)
            print(
                json.dumps(
                    {"mode_state_completed": state["state_id"], "rollouts": len(rows)}
                ),
                flush=True,
            )
            if progress:
                progress(state_id=state["state_id"], rollouts=len(rows))
    finally:
        restore_random_state(outside)
    assert digest_tree(algorithm.get_state()) == before
    differences = []
    for state in states:
        data = [r for r in rows if r["state_id"] == state["state_id"]]
        means = [
            np.mean([r["return"] for r in data if r["mode"] == m]) for m in range(3)
        ]
        dmc = [
            next(
                r["return"] for r in data if r["mode"] == 2 and r["replication"] == rep
            )
            - next(
                r["return"] for r in data if r["mode"] == 1 and r["replication"] == rep
            )
            for rep in MODE_REPLICATIONS
        ]
        differences.append(
            {
                "state_id": state["state_id"],
                "month": state["month"],
                "soc_fraction": state["soc_fraction"],
                "idle_return": float(means[0]),
                "charge_return": float(means[1]),
                "discharge_return": float(means[2]),
                "discharge_minus_charge": float(means[2] - means[1]),
                "difference_same_sign_all_replicates": bool(
                    min(dmc) > 0 or max(dmc) < 0
                ),
                "argmax": state["actor_argmax"],
                "best_sample_mode": int(np.argmax(means)),
                "conditional_mode_regret": float(
                    max(means) - means[state["actor_argmax"]]
                ),
                "probability_weighted_regret": float(max(means) - sum(
                    state[f"prob_{name}"] * means[i]
                    for i, name in enumerate(("idle", "charge", "discharge"))
                )),
                "uniform_mode_regret": float(max(means) - np.mean(means)),
            }
        )
    write_csv(target / "states.csv", states)
    write_csv(target / "mode-differences.csv", differences)
    common_targets = [
        np.mean(
            [
                r["return"]
                for r in rows
                if r["state_id"] == s["state_id"] and r["mode"] == -1
            ]
        )
        for s in states
    ]
    result = {
        "states": len(states),
        "rollouts": len(rows),
        "physical_periods": sum(r["periods"] for r in rows),
        "mean_conditional_mode_regret": float(
            np.mean([r["conditional_mode_regret"] for r in differences])
        ),
        "mean_probability_weighted_regret": float(
            np.mean([r["probability_weighted_regret"] for r in differences])
        ),
        "mean_uniform_mode_regret": float(np.mean([r["uniform_mode_regret"] for r in differences])),
        "sample_argmax_agreement": float(
            np.mean([r["argmax"] == r["best_sample_mode"] for r in differences])
        ),
        "probability_ranges": {
            name: [
                min(r[f"prob_{name}"] for r in states),
                max(r[f"prob_{name}"] for r in states),
            ]
            for name in ("idle", "charge", "discharge")
        },
        "training_state_unchanged": True,
        "common_state_critic": error_statistics(
            [s["prediction"] for s in states], common_targets
        ),
        "counterfactual_rollouts": sum(r["mode"] >= 0 for r in rows),
        "policy_continuation_rollouts": sum(r["mode"] == -1 for r in rows),
        "scope": "First-step conditional-power-mean mode interventions; no optimal-Q or policy-gain claim",
        "snapshot_source": str(MODE_SOURCE),
        "snapshot_hashes_verified": True,
    }
    write_json(target / "summary.json", result)
    return result
