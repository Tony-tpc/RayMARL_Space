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

from experiments.sn_energy_market.training_env import MarketEnvFactory


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Run a complete small training lifecycle"
    )
    parser.add_argument("--config-dir", type=Path)
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
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args(argv)


def experiment_settings(args):
    base = (
        EnvironmentConfig.from_directory(args.config_dir)
        if args.config_dir
        else EnvironmentConfig()
    )
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
    config = replace(base, system=system, market=market).validate()
    if config.system.episode_mode != "finite":
        raise ValueError(
            "This episode experiment requires finite profiles; use the adapter for continuous sampling"
        )
    if args.evaluate_only and args.resume is None:
        raise ValueError("--evaluate-only requires --resume")
    defaults = {
        "outer_rounds": (1, 20),
        "inner_updates": (1, 5),
        "response_updates": (1, 5),
        "candidates": (1, 2),
        "proposal_updates": (1, 1),
        "deviation_updates": (1, 2),
        "rollout_steps": (8, 128),
        "epochs": (1, 4),
        "minibatch_size": (8, 64),
        "hidden_dim": (16, 64),
    }
    budget = {
        key: getattr(args, key)
        if getattr(args, key) is not None
        else values[not args.smoke]
        for key, values in defaults.items()
    }
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
    return config, budget, splits


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
    paths.extend([Path(__file__), Path(__file__).with_name("training_env.py")])
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
                "sha256": (p := factory(seed).engine.profiles).fingerprint(),
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
        "profiles": profiles,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "resume": args.resume,
        "normalization": "fixed signed_log1p observations; per-batch advantage normalization",
        "candidate_protocol": "same reference Consumer snapshot, response budget and validation scenarios",
        "test_protocol": "frozen final policies evaluated only after selection",
    }


def main(argv=None):
    args = arguments(argv)
    environment, budget, splits = experiment_settings(args)
    output = (
        args.output
        or ROOT
        / "outputs/snmarket-training"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / ".tmp" / "market-training"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(temporary)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    secret = ROOT / ".ray" / "auth_token"
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
    factory = MarketEnvFactory(environment, seed_splits=splits)
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
                validation_seeds=splits["validation"],
                test_seeds=splits["test"],
            )
            .debugging(seed=args.seed)
            .rl_module(model_config={"hidden_dim": budget["hidden_dim"]})
            .learners(num_learners=0, num_gpus_per_learner=int(args.device == "cuda"))
            .training(
                rollout_length=budget["rollout_steps"],
                inner_updates=budget["inner_updates"],
                response_updates=budget["response_updates"],
                uc_candidates=budget["candidates"],
                uc_updates_per_candidate=budget["proposal_updates"],
                deviation_updates=budget["deviation_updates"],
                num_epochs=budget["epochs"],
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
        evaluation = algorithm.evaluate_pair()
        deviations = (
            {"searched": False, "reason": "frozen_evaluation"}
            if args.evaluate_only
            else algorithm.last_round["consumer_deviations_selected_pair"]
        )
        write_json(output / "held_out_evaluation.json", evaluation)
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
