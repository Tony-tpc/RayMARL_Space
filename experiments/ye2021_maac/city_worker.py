"""Archived sidecar: uses unmodified CityLearn physics and official MARL reward."""
import contextlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def environment(config):
    from citylearn.citylearn import CityLearnEnv
    from citylearn.data import DataSet
    from citylearn.reward_function import MARL
    class WorkspaceDataSet(DataSet):
        @property
        def cache_directory(self):
            path = ROOT/"datasets"/"citylearn"
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
    schema = config.get("schema")
    if not schema:
        schema = WorkspaceDataSet(tag="v3.0.2").get_dataset("citylearn_challenge_2022_phase_all_plus_evs")
    parsed = json.loads(Path(schema).read_text())
    inactive = [k for k in parsed["observations"] if "predicted" in k]
    windows = {"train": range(36), "validation": range(36, 44), "test": range(44, 52), "probe": range(1)}
    weeks = config.get("weeks", windows[config.get("split", "train")])
    episode_size = config.get("episode_steps", 168)
    return CityLearnEnv(schema, buildings=config.get("buildings", ["Building_1", "Building_2", "Building_4"]),
        central_agent=False, interface="flat", render_mode="none", reward_function=MARL,
        inactive_observations=inactive,
        inactive_actions=["deferrable_appliance"] if config.get("track") == "continuous" else [],
        episode_time_steps=[(w*168, w*168+episode_size-1) for w in weeks],
        random_episode_split=config.get("split", "train") == "train", random_seed=config.get("seed", 0))


def main():
    env = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if env is None:
                    env = environment(request["config"])
                command = request["command"]
                if command == "describe":
                    result = {"observation_names": env.observation_names, "action_names": env.action_names,
                        "action_low": [s.low for s in env.action_space],
                        "action_high": [s.high for s in env.action_space],
                        "buildings": [b.name for b in env.buildings]}
                elif command == "reset":
                    obs, info = env.reset(seed=request.get("seed"))
                    result = {"observations": obs}
                elif command == "step":
                    obs, reward, terminated, truncated, info = env.step(request["actions"])
                    result = {"observations": obs, "rewards": reward, "terminated": bool(terminated),
                              "truncated": bool(truncated)}
                elif command == "evaluate":
                    frame = env.evaluate_v2(include_business_as_usual=False)
                    result = frame.replace({np.nan: None}).to_dict(orient="records")
                elif command == "close":
                    env.close()
                    result = None
                else:
                    raise ValueError(command)
            print(json.dumps({"result": result}, default=serialize), flush=True)
            if command == "close":
                return
        except Exception:
            print(json.dumps({"error": traceback.format_exc()}), flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
