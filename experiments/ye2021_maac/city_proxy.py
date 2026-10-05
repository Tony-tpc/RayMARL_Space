"""Archived Gymnasium/RLlib proxy; no CityLearn imports in the training venv."""
from collections import deque
import json
from pathlib import Path
import queue
import subprocess
import threading
import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

ROOT = Path(__file__).resolve().parents[2]


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class CityLearnProxy(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.worker_config = dict(config or {})
        command = [str(ROOT/".venv-citylearn"/"Scripts"/"python.exe"), "-u",
                   str(Path(__file__).with_name("city_worker.py"))]
        logdir = ROOT/"outputs"/"city-worker"
        logdir.mkdir(parents=True, exist_ok=True)
        self.log = (logdir/f"{threading.get_ident()}-{id(self)}.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.responses = queue.Queue()
        def reader():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=reader, daemon=True).start()
        metadata = self.request("describe", timeout=180)
        self.possible_agents = metadata["buildings"]
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.possible_agents)
        self.names = metadata["observation_names"]
        self.action_names = metadata["action_names"]
        self.continuous, self.binary = [], []
        self.action_spaces, self.observation_spaces = {}, {}
        self.history = {a: deque(maxlen=48) for a in self.agents}
        for i, aid in enumerate(self.agents):
            c = [j for j, n in enumerate(self.action_names[i]) if not n.startswith("deferrable_appliance")]
            b = [j for j, n in enumerate(self.action_names[i]) if n.startswith("deferrable_appliance")]
            self.continuous.append(c)
            self.binary.append(b)
            box = gym.spaces.Box(np.asarray(metadata["action_low"][i], np.float32)[c],
                                 np.asarray(metadata["action_high"][i], np.float32)[c], dtype=np.float32)
            self.action_spaces[aid] = (gym.spaces.Dict({"continuous": box,
                "binary": gym.spaces.MultiBinary(len(b))}) if b else box)
            self.observation_spaces[aid] = gym.spaces.Dict({
                "current": gym.spaces.Box(-np.inf, np.inf, (len(self.names[i]),), np.float32),
                "history": gym.spaces.Box(-np.inf, np.inf, (48, 3), np.float32),
                "history_mask": gym.spaces.Box(0, 1, (48,), np.float32)})
            if self.worker_config.get('history_enabled') is False:
                self.observation_spaces[aid]=self.observation_spaces[aid]['current']

    def request(self, command, timeout=30, **kwargs):
        self.process.stdin.write(json.dumps({"command": command, "config": self.worker_config, **kwargs},
                                            default=json_value)+"\n")
        self.process.stdin.flush()
        try:
            response = self.responses.get(timeout=timeout)
        except queue.Empty as exc:
            self.process.kill()
            raise TimeoutError(f"CityLearn {command} timed out") from exc
        if response is None:
            raise RuntimeError(f"CityLearn worker exited with {self.process.poll()}")
        payload = json.loads(response)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return payload["result"]

    def _observations(self, raw):
        result = {}
        for aid, names, values in zip(self.agents, self.names, raw):
            values = np.asarray(values, np.float32)
            # Fixed physical scales; no fit to held-out data or future statistics.
            scales = np.array([8760 if "time_step" in n else 24 if ("hour" in n or "departure_time" in n
                              or "arrival_time" in n or "duration_steps" in n) else 12 if n == "month"
                              else 8 if n=="day_type" else 40 if "temperature" in n
                              else 100 if ("capacity" in n or "humidity" in n)
                              else 1000 if "irradiance" in n
                              else 10 if ("load" in n or "generation" in n or "energy_kwh" in n)
                              else 1 for n in names], np.float32)
            current = values/scales
            row = [values[names.index(n)]/(10 if n != "electricity_pricing" else 1)
                   for n in ("electricity_pricing", "solar_generation", "non_shiftable_load")]
            self.history[aid].append(row)
            history = np.zeros((48, 3), np.float32)
            mask = np.zeros(48, np.float32)
            history[:len(self.history[aid])] = self.history[aid]
            mask[:len(self.history[aid])] = 1
            result[aid] = ({"current": current, "history": history, "history_mask": mask}
                          if self.worker_config.get('history_enabled',True) else current)
        return result

    def reset(self, *, seed=None, options=None):
        for history in self.history.values():
            history.clear()
        result = self.request("reset", seed=seed, timeout=60)
        return self._observations(result["observations"]), {a: {} for a in self.agents}

    def step(self, actions):
        encoded = []
        for i, aid in enumerate(self.agents):
            action = actions[aid]
            vector = np.zeros(len(self.action_names[i]), np.float32)
            if self.binary[i]:
                vector[self.continuous[i]] = action["continuous"]
                vector[self.binary[i]] = action["binary"]
            else:
                vector[self.continuous[i]] = action
            encoded.append(vector)
        result = self.request("step", actions=encoded)
        rewards = dict(zip(self.agents, result["rewards"]))
        term = {a: result["terminated"] for a in [*self.agents, "__all__"]}
        trunc = {a: result["truncated"] for a in [*self.agents, "__all__"]}
        return self._observations(result["observations"]), rewards, term, trunc, {a: {} for a in self.agents}

    def evaluate(self):
        return self.request("evaluate", timeout=60)

    def close(self):
        if hasattr(self, "process") and self.process.poll() is None:
            try:
                self.request("close", timeout=5)
                self.process.wait(timeout=5)
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                self.process.kill()
        if hasattr(self, "log"):
            self.log.close()
