"""Minimal MAPPO training; provide class aaa in aaa.py."""

import ray

from raymarl import MAPPOConfig


def main():
    from aaa import aaa

    ray.init()
    algo = None
    try:
        algo = (
            MAPPOConfig()
            .environment(aaa)
            .env_runners(num_env_runners=0)
            .learners(num_learners=0, num_gpus_per_learner=0)
            .build_algo()
        )
        for _ in range(100):
            result = algo.train()
            print(result.get("env_runners", {}).get("episode_return_mean"))
        print(algo.save_to_path())
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
