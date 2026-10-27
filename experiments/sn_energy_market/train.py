"""Current energy-market Stackelberg training, matched validation and held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RLAlgo/src"))
sys.path.insert(0, str(ROOT / "environments/sn_energy_market/src"))

from snmarket import EnvironmentConfig

from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)
from experiments.sn_energy_market.training_env import MarketEnvFactory


def model_settings(hidden_dim):
    """Experiment priors and value units; physical actions remain freely sampled."""
    return {
        "hidden_dim": hidden_dim,
        "uc_initial_mean": [0.95, 0.02, 0.2, 0.2, 0.5, 0.5],
        "uc_initial_std": [0.08, 0.05, 0.2, 0.2, 0.2, 0.2],
        "uc_head_scale": 0.01,
        "value_scale": {"uc": 20.0, "consumer": 10.0},
    }


def inventory_valued_config(config):
    """Fixed reference inventory value in stored-kWh units, shared by all policies.

    Uses the experiment's 0.35 off-peak reference and accounts for discharge
    efficiency and output throughput cost. Explicit positive values are kept.
    """
    if config.rewards.terminal_energy_value_per_kwh > 0:
        return config
    value = config.system.ess_discharge_efficiency * max(
        0.0, 0.35 - config.rewards.ess_throughput_rate_per_kwh
    )
    return replace(config, rewards=replace(config.rewards, terminal_energy_value_per_kwh=value)).validate()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Run a complete small training lifecycle"
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Cleaned profile manifest with chronological train/validation/test splits",
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Verify data and experiment contracts without importing Ray or training",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Load only training/validation data and reserve the test set for final evaluation",
    )
    parser.add_argument("--consumers", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--deadline", type=int)
    parser.add_argument("--outer-rounds", type=int)
    parser.add_argument("--inner-updates", type=int)
    parser.add_argument("--response-updates", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--proposal-updates", type=int)
    parser.add_argument("--deviation-updates", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--critic-epochs", type=int)
    parser.add_argument("--critic-lambda", type=float, default=1.0)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args(argv)


def load_dataset(args):
    if not args.dataset:
        return None
    splits = (
        ("train", "validation") if args.skip_test else ("train", "validation", "test")
    )
    return ProfileDataset(args.dataset, splits=splits)


def experiment_settings(args, dataset=None):
    base = (
        EnvironmentConfig.from_directory(args.config_dir)
        if args.config_dir
        else EnvironmentConfig()
    )
    dataset = dataset or load_dataset(args)
    count = (
        args.consumers
        if args.consumers is not None
        else (2 if args.smoke else base.system.num_consumers)
    )
    horizon = (
        args.horizon
        if args.horizon is not None
        else 8
        if args.smoke
        else base.system.horizon_steps
        if args.config_dir
        else 288
    )
    deadline = (
        args.deadline
        if args.deadline is not None
        else (2 if args.smoke else base.market.shift_deadline_steps)
    )
    system = replace(base.system, num_consumers=count, horizon_steps=horizon)
    if args.smoke:
        system = replace(system, start_datetime="2026-01-01T17:00:00+08:00")
    market = replace(base.market, shift_deadline_steps=deadline)
    config = (
        dataset.environment_config(base if args.config_dir else None)
        if dataset is not None
        else replace(base, system=system, market=market).validate()
    )
    if dataset is not None:
        if args.horizon is not None:
            raise ValueError(
                "Dataset episodes use complete calendar months; --horizon cannot truncate them"
            )
        if args.consumers is not None and args.consumers != len(dataset.household_ids):
            raise ValueError(
                "--consumers must match the dataset's fixed household identities"
            )
        if args.deadline is not None:
            config = replace(
                config,
                market=replace(config.market, shift_deadline_steps=args.deadline),
            ).validate()
    config = inventory_valued_config(config)
    if config.system.episode_mode != "finite":
        raise ValueError(
            "This episode experiment requires finite profiles; use the adapter for continuous sampling"
        )
    if args.evaluate_only and args.resume is None:
        raise ValueError("--evaluate-only requires --resume")
    if args.validate_data and dataset is None:
        raise ValueError("--validate-data requires --dataset")
    defaults = {
        "outer_rounds": (1, 20),
        "inner_updates": (1, 5),
        "response_updates": (1, 5),
        "candidates": (1, 2),
        "proposal_updates": (1, 1),
        "deviation_updates": (1, 2),
        "rollout_steps": (8, 128),
        "epochs": (1, 4),
        "critic_epochs": (1, 4),
        "minibatch_size": (8, 64),
        "hidden_dim": (16, 64),
    }
    budget = {
        key: getattr(args, key)
        if getattr(args, key) is not None
        else values[not args.smoke]
        for key, values in defaults.items()
    }
    budget.update(gamma=0.99, lambda_gae=0.97, critic_lambda=args.critic_lambda)
    if not 0 <= args.critic_lambda <= 1:
        raise ValueError("--critic-lambda must lie in [0, 1]")
    if dataset is not None and args.rollout_steps is None and not args.smoke:
        budget["rollout_steps"] = 192
    if not args.evaluate_only and budget["outer_rounds"] < 1:
        raise ValueError("--outer-rounds must be positive for training")
    train_count, evaluation_count = (2, 2) if args.smoke else (8, 8)
    splits = {
        "train": tuple(range(args.seed, args.seed + train_count)),
        "validation": tuple(
            range(args.seed + 10_000, args.seed + 10_000 + evaluation_count)
        ),
        "test": tuple(range(args.seed + 20_000, args.seed + 20_000 + evaluation_count)),
    }
    if dataset is not None:
        splits = DatasetEnvFactory(
            dataset, config, seed=args.seed
        ).seed_splits
    if args.skip_test:
        splits.pop("test", None)
    return config, budget, splits


def scenario_factory(args, config, splits, dataset=None):
    if args.dataset:
        return DatasetEnvFactory(
            dataset or load_dataset(args),
            config,
            seed=args.seed,
        )
    return MarketEnvFactory(config, seed_splits=splits)


def write_json(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported report value {type(item).__name__}")

    Path(path).write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, allow_nan=False, default=encode
        ),
        encoding="utf-8",
    )


def provenance(config, budget, splits, factory, args, learning_contract):
    paths = [ROOT / "environments/sn_energy_market/docs/v9/modeling_complete_zh.md"]
    for directory in (
        "environments/sn_energy_market/src/snmarket",
        "RLAlgo/src/raymarl/algorithms/stackelberg",
    ):
        paths.extend(sorted((ROOT / directory).glob("*.py")))
    paths.extend(
        [
            Path(__file__),
            Path(__file__).with_name("training_env.py"),
            Path(__file__).with_name("profile_dataset.py"),
        ]
    )
    sources = {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    profiles = {
        split: [
            {
                "seed": seed,
                "sha256": (p := (
                    factory.profiles_for_seed(seed) if args.dataset
                    else factory(seed).engine.profiles
                )).fingerprint(),
                "metadata": p.metadata,
            }
            for seed in seeds
        ]
        for split, seeds in splits.items()
    }
    return {
        "git_commit": commit.stdout.strip(),
        "source_sha256": sources,
        "environment": config.to_dict(),
        "budget": budget,
        "learning_contract": learning_contract,
        "seed_splits": splits,
        "scenario_contract": factory.get_contract(),
        "profiles": profiles,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "resume": args.resume,
        "normalization": "fixed signed_log1p observations; per-batch advantage normalization",
        "candidate_protocol": "same reference Consumer snapshot, response budget and validation scenarios",
        "test_protocol": (
            "reserved; no test profiles loaded or evaluated"
            if args.skip_test
            else "frozen final policies evaluated only after selection"
        ),
    }


def main(argv=None):
    args = arguments(argv)
    dataset = load_dataset(args)
    environment, budget, splits = experiment_settings(args, dataset)
    factory = scenario_factory(args, environment, splits, dataset)
    if args.validate_data:
        report = {
            "status": "data_contract_validated",
            "training_started": False,
            "test_reserved": args.skip_test,
            "environment": environment.to_dict(),
            "budget": budget,
            "scenario_contract": factory.get_contract(),
            "splits": {
                split: [
                    {
                        "seed": seed,
                        "periods": len(profiles),
                        "start_datetime": profiles.start_datetime.isoformat(),
                        "sha256": profiles.fingerprint(),
                        "metadata": profiles.metadata,
                    }
                    for seed in seeds
                    for profiles in [
                        factory.profiles_for_seed(seed) if args.dataset
                        else factory(seed).engine.profiles
                    ]
                ]
                for split, seeds in splits.items()
            },
        }
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / "data_contract.json", report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "training_started": False,
                    "monthly_scenarios": {
                        key: len(value) for key, value in splits.items()
                    },
                },
                ensure_ascii=False,
            )
        )
        return report
    output = (
        args.output
        or Path("D:/Codex/snmarket-training")
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "runtime"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(temporary)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    secret = temporary / "auth_token"
    secret.parent.mkdir(exist_ok=True)
    if not secret.exists():
        secret.write_text(secrets.token_hex(32), encoding="utf-8")
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(secret)

    import ray
    import torch

    from raymarl.algorithms.stackelberg import StackelbergConfig

    torch.set_num_threads(args.torch_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(temporary / "ray"),
    )
    algorithm = None
    try:
        configuration = (
            StackelbergConfig()
            .scenarios(
                factory,
                train_seeds=splits["train"],
                train_seed_groups=factory.training_seed_groups if args.dataset else None,
                validation_seeds=splits["validation"],
                test_seeds=splits.get("test", ()),
            )
            .debugging(seed=args.seed)
            .rl_module(model_config=model_settings(budget["hidden_dim"]))
            .learners(num_learners=0, num_gpus_per_learner=int(args.device == "cuda"))
            .training(
                execution_workers=4 if args.dataset and len(factory.training_seed_groups) == 4 else 0,
                gamma=budget["gamma"],
                lambda_gae=budget["lambda_gae"],
                critic_lambda=budget["critic_lambda"],
                rollout_length=budget["rollout_steps"],
                inner_updates=budget["inner_updates"],
                response_updates=budget["response_updates"],
                uc_candidates=budget["candidates"],
                uc_updates_per_candidate=budget["proposal_updates"],
                deviation_updates=budget["deviation_updates"],
                num_epochs=budget["epochs"],
                critic_epochs=budget["critic_epochs"],
                minibatch_size=budget["minibatch_size"],
                storage_path=str(output),
                checkpoint_interval=1,
            )
        )
        algorithm = configuration.build_algo()
        if args.resume:
            algorithm.restore_from_path(str(args.resume.resolve()))
        write_json(
            output / "experiment.json",
            provenance(
                environment,
                budget,
                splits,
                factory,
                args,
                configuration.learning_contract(),
            ),
        )
        if not args.evaluate_only:
            for _ in range(budget["outer_rounds"]):
                result = algorithm.train()
                write_json(
                    output / f"round-{algorithm.iteration:04d}.json",
                    {
                        "training_iteration": algorithm.iteration,
                        "round": algorithm.last_round,
                        "learners": result.get("learners", {}),
                    },
                )
                print(
                    json.dumps(
                        {"outer_round": algorithm.iteration, "status": "completed"}
                    ),
                    flush=True,
                )
                if algorithm.last_round["stop_recommended"]:
                    break
        evaluation_split = "validation" if args.skip_test else "test"
        evaluation = algorithm.evaluate_pair(splits[evaluation_split])
        deviations = (
            {"searched": False, "reason": "frozen_evaluation"}
            if args.evaluate_only
            else algorithm.last_round["consumer_deviations_selected_pair"]
        )
        evaluation_file = (
            "validation_evaluation.json"
            if args.skip_test
            else "held_out_evaluation.json"
        )
        write_json(output / evaluation_file, evaluation)
        write_json(output / "deviation_search.json", deviations)
        checkpoint = algorithm.save_to_path(str(output / "final-checkpoint"))
        algorithm.export_actors(str(output / "actors"))
        write_json(
            output / "result.json",
            {
                "status": "completed",
                "outer_rounds": algorithm.iteration,
                "checkpoint": checkpoint,
                "evaluation": evaluation,
                "evaluation_split": evaluation_split,
                "test_reserved": args.skip_test,
                "deviation_search": deviations,
            },
        )
        print(f"训练与评价产物：{output}", flush=True)
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
