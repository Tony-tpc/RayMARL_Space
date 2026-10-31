"""Validate real-data episodes and policy sampling without optimizer updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "environments/sn_energy_market/src"))
sys.path.insert(0, str(ROOT / "RLAlgo/src"))

from snmarket.action import map_consumer_controls

from experiments.sn_energy_market.validation_helpers import (
    check_observations,
    check_record,
    compare_oracle,
    json_safe,
)

_WORKER_FACTORY = None


def initialize_worker(factory):
    global _WORKER_FACTORY
    _WORKER_FACTORY = factory


def validate_month(seed, split):
    return validate_episode(_WORKER_FACTORY, seed, split)


def validate_episode(factory, seed, split):
    """Use the training adapter, with independent accounting checks every step."""
    env = factory(seed)
    env.reset(seed)
    rng = np.random.default_rng(seed)
    coverage, maximums, totals = Counter(), {}, Counter()
    periods = env.config.system.horizon_steps
    for index in range(periods):
        before = env.engine.state
        controls = rng.uniform(-0.25, 1.25, 6)
        if index % 48 == 0:
            controls[:] = 0
        elif index % 48 == 1:
            controls[:] = 1
        decision = env.publish(controls)
        actions = {
            key: map_consumer_controls(rng.random(4), bounds)
            for key, bounds in env.engine.local_bounds().items()
        }
        check_observations(
            env.engine,
            before,
            decision.uc_observation,
            decision.consumer_observations,
            actions,
            maximums,
        )
        assert env.engine.state == before
        saved = env.get_state() if index == 49 else None
        transition = env.step(actions)
        record = env.engine.latest_record
        check_record(
            env.engine, before, record.uc_action, actions, record, coverage, maximums
        )
        np.testing.assert_array_equal(
            decision.uc_action, np.asarray(tuple(vars(record.uc_action).values()))
        )
        assert transition.terminated == (index == periods - 1)
        assert not transition.data_exhausted
        if index % 48 == 24 or index in (0, periods - 1):
            residuals = compare_oracle(env.engine, before, record)
            for name, value in residuals.items():
                maximums[f"oracle_{name}"] = max(
                    maximums.get(f"oracle_{name}", 0.0), value
                )
            coverage["oracle_comparisons"] += 1
        if saved is not None:
            after = env.engine.state
            env.set_state(saved)
            repeated = env.step(actions)
            assert repeated.rewards == transition.rewards
            assert repeated.metrics == transition.metrics
            assert env.engine.latest_record == record and env.engine.state == after
            coverage["snapshot_replays"] += 1
        totals.update(transition.metrics)
    return {
        "split": split,
        "seed": seed,
        "start_datetime": env.engine.profiles.start_datetime.isoformat(),
        "end_datetime": env.engine.state.timestamp.isoformat(),
        "physical_periods": periods,
        "consumers": len(env.consumer_ids),
        "profile_sha256": env.engine.profiles.fingerprint(),
        "coverage": dict(coverage),
        "maximums": maximums,
        "totals": dict(totals),
    }


def validate_sampling(factory, budget, splits):
    """Exercise the actual RLModule/EnvRunner path; weights remain untouched."""
    import torch
    from raymarl.algorithms.stackelberg import StackelbergConfig
    from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner

    torch.set_num_threads(1)
    config = (
        StackelbergConfig()
        .scenarios(
            factory,
            train_seeds=splits["train"],
            validation_seeds=splits["validation"],
            test_seeds=splits["test"],
        )
        .rl_module(model_config={"hidden_dim": 16})
        .training(gamma=budget["gamma"], lambda_gae=budget["lambda_gae"])
    )
    config.prepare_policies()
    runner = StackelbergEnvRunner(config=config)
    before = {
        (agent, key): value.detach().clone()
        for agent in runner.agent_ids
        for key, value in runner.module[agent].state_dict().items()
    }
    batches = [runner.collect(64), runner.collect(64)]
    assert runner.env.engine.state.period_id == 128
    for batch in batches:
        assert batch.count == 64
        for key, rows in batch.policy_batches.items():
            assert rows["actions"].shape == (64, 6 if key == "uc" else 4)
            for column in ("actions", "action_logp", "advantages", "value_targets"):
                assert np.isfinite(rows[column]).all(), (key, column)
            assert not rows["terminateds"].any()
            assert rows["truncateds"][-1]
    for agent in runner.agent_ids:
        for key, value in runner.module[agent].state_dict().items():
            torch.testing.assert_close(value, before[(agent, key)], atol=0, rtol=0)
    result = {
        "physical_periods": 128,
        "batches": 2,
        "agents": len(runner.agent_ids),
        "uc_observation_dim": runner.env.uc_observation_dim,
        "consumer_observation_dim": runner.env.consumer_observation_dim,
        "weights_unchanged": True,
        "optimizer_updates": 0,
        "gamma": config.gamma,
        "lambda_gae": config.lambda_gae,
    }
    runner.stop()
    return result


def validate(factory, budget, splits, output, *, sampling=True, workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episodes, coverage, maximums = [], Counter(), {}
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_worker, initargs=(factory,)
    ) as executor:
        futures = {
            executor.submit(validate_month, seed, split): (split, seed)
            for split, seeds in splits.items()
            for seed in seeds
        }
        for future in as_completed(futures):
            split, seed = futures[future]
            try:
                episode = future.result()
            except Exception as exc:
                failure = {
                    "split": split,
                    "seed": seed,
                    "error": repr(exc),
                    "factory_contract": factory.get_contract(),
                }
                (output / "failure.json").write_text(
                    json.dumps(json_safe(failure), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise
            episodes.append(episode)
            (output / f"{split}-{seed}.json").write_text(
                json.dumps(
                    json_safe(episode), ensure_ascii=False, indent=2, allow_nan=False
                ),
                encoding="utf-8",
            )
            coverage.update(episode["coverage"])
            for name, value in episode["maximums"].items():
                maximums[name] = max(maximums.get(name, 0.0), value)
            print(
                json.dumps(
                    {
                        "split": split,
                        "seed": seed,
                        "periods": episode["physical_periods"],
                    }
                ),
                flush=True,
            )
    episodes.sort(key=lambda item: item["start_datetime"])
    paths = list(Path(__file__).parent.glob("*.py"))
    paths.extend((ROOT / "environments/sn_energy_market/src/snmarket").glob("*.py"))
    paths.extend((ROOT / "RLAlgo/src/raymarl/algorithms/stackelberg").glob("*.py"))
    report = {
        "status": "passed",
        "no_training": True,
        "physical_periods": sum(row["physical_periods"] for row in episodes),
        "consumer_periods": sum(
            row["physical_periods"] * row["consumers"] for row in episodes
        ),
        "episodes": episodes,
        "coverage": dict(coverage),
        "maximums": maximums,
        "factory_contract": factory.get_contract(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        },
        "sampling": validate_sampling(factory, budget, splits) if sampling else None,
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main(argv=None):
    from experiments.sn_energy_market.profile_dataset import (
        DatasetEnvFactory,
        ProfileDataset,
    )
    from experiments.sn_energy_market.settings import inventory_valued_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("D:/Codex/ausgrid-validation")
    )
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    dataset = ProfileDataset(args.dataset, splits=("train", "validation", "test"))
    config = inventory_valued_config(dataset.environment_config())
    factory = DatasetEnvFactory(dataset, config, seed=42)
    splits = factory.seed_splits
    budget = {"gamma": 0.997, "lambda_gae": 0.97}
    report = validate(
        factory,
        budget,
        splits,
        args.output,
        sampling=not args.skip_sampling,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "physical_periods": report["physical_periods"],
                "maximums": report["maximums"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
