"""Small deterministic interface fixture, not an energy simulator."""

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class InterfaceGame(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.hybrid = config.get("hybrid", False)
        self.horizon = config.get("horizon", 4)
        self.possible_agents = ["a", "b"]
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.possible_agents)
        self.observation_spaces = {
            "a": gym.spaces.Box(-1, 1, (2,), np.float32),
            "b": gym.spaces.Box(-1, 1, (3,), np.float32),
        }
        action = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.action_spaces = {
            "a": gym.spaces.Dict({"continuous": action, "binary": gym.spaces.MultiBinary(1)})
            if self.hybrid
            else action,
            "b": action,
        }
        self.t = 0

    def _obs(self):
        return {
            a: np.full(s.shape, self.t / self.horizon, np.float32) for a, s in self.observation_spaces.items()
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        a0 = float(np.asarray(actions["a"]["continuous"] if self.hybrid else actions["a"])[0])
        a1 = float(np.asarray(actions["b"])[0])
        self.t += 1
        reward = {"a": -((a0 - 0.4) ** 2) + 0.1 * a1, "b": -((a1 + 0.3) ** 2) - 0.1 * a0}
        end = self.t >= self.horizon
        return (
            self._obs(),
            reward,
            {"a": end, "b": end, "__all__": end},
            {"a": False, "b": False, "__all__": False},
            {a: {} for a in self.agents},
        )
