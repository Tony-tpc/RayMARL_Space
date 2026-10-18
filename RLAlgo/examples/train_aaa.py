"""Minimal MAPPO training; provide class aaa in environments/aaa.py."""

import os
import secrets
import sys
from pathlib import Path

import ray


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "RLAlgo" / "src"), str(root)]
    from environments.aaa import aaa
    from raymarl import MAPPOConfig

    if not os.environ.get("RAY_AUTH_TOKEN") and not os.environ.get("RAY_AUTH_TOKEN_PATH"):
        token = root / ".ray" / "auth_token"
        token.parent.mkdir(exist_ok=True)
        if not token.exists():
            token.write_text(secrets.token_hex(32), encoding="ascii")
        os.environ["RAY_AUTH_TOKEN_PATH"] = str(token)
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    ray.init(
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(root / ".tmp" / "train-aaa-ray"),
    )
    algo = None
    try:
        algo = (
            MAPPOConfig()
            .environment(aaa)
            .env_runners(num_env_runners=0)
            .learners(num_learners=0, num_gpus_per_learner=0)
            .training(storage_path=str(root / "outputs"))
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
