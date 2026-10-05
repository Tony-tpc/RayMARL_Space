"""Run the RLlib documentation's Taxi/PPO example on a local Windows machine."""

import argparse
import json
from pathlib import Path

from ray_local import configure

configure()

import ray
import torch
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.train._internal.storage import StorageContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--address", default=None, help="Use 'auto' to join a running Ray node")
    parser.add_argument("--gpus", type=int, choices=(0, 1), default=None,
                        help="Defaults to 1 when CUDA is available; use 0 for CPU")
    parser.add_argument("--env-runners", type=int, choices=(0, 1, 2), default=0,
                        help="Use 0 to sample locally and reduce Windows host memory use")
    args = parser.parse_args()
    if args.gpus is None:
        args.gpus = int(torch.cuda.is_available())
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.gpus and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")

    output = Path(__file__).resolve().parent / "artifacts"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    init_kwargs = {"address": args.address} if args.address else {
        "num_cpus": 4, "include_dashboard": False,
        "_node_ip_address": "127.0.0.1",
        "_temp_dir": str(Path(__file__).resolve().parent / ".tmp" / "ray"),
    }
    ray.init(**init_kwargs)
    algo = None
    try:
        config = (
            PPOConfig()
            .environment("Taxi-v3")
            .framework("torch")
            .env_runners(
                num_env_runners=args.env_runners,
                rollout_fragment_length=512 // max(1, args.env_runners),
                env_to_module_connector=lambda env, spaces=None, device=None: FlattenObservations(),
            )
            .learners(num_learners=0, num_gpus_per_learner=args.gpus)
            .training(train_batch_size_per_learner=512, minibatch_size=128, num_epochs=2)
            .evaluation(
                evaluation_num_env_runners=0,
                evaluation_duration=2,
                evaluation_duration_unit="episodes",
            )
            .debugging(seed=42)
        )
        # Ray 2.59 removed logger_creator; use its storage context to keep all
        # outputs inside this workspace instead of writing ~/ray_results.
        storage = StorageContext(
            storage_path=output.as_posix(),
            experiment_dir_name="taxi-ppo",
            trial_dir_name="local",
        )
        algo = PPO(config=config, storage=storage)
        learner_module = algo.learner_group._learner.module[DEFAULT_MODULE_ID]
        learner_device = str(next(learner_module.parameters()).device)
        if args.gpus and not learner_device.startswith("cuda"):
            raise RuntimeError(f"Expected a CUDA learner, got {learner_device}")
        print(f"Learner device: {learner_device}", flush=True)
        training = []
        for i in range(args.iterations):
            result = algo.train()
            runners = result.get("env_runners", {})
            summary = {
                "iteration": i + 1,
                "episode_return_mean": runners.get("episode_return_mean"),
                "num_env_steps_sampled_lifetime": result.get("num_env_steps_sampled_lifetime"),
            }
            training.append(summary)
            print(json.dumps(summary, default=str), flush=True)
        evaluation = algo.evaluate()
        checkpoint = algo.save_to_path(str(output / "checkpoint"))
        report = {
            "ray": ray.__version__, "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "learner_device": learner_device,
            "gpu": torch.cuda.get_device_name(0) if args.gpus else None,
            "training": training,
            "evaluation": evaluation,
            "checkpoint": str(checkpoint),
        }
        (output / "quickstart-result.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"Checkpoint: {checkpoint}", flush=True)
        print(f"Report: {output / 'quickstart-result.json'}", flush=True)
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
