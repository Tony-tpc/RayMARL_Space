"""Uniform replay of complete synchronized environment transitions."""

import copy

import numpy as np


class JointReplay:
    def __init__(self, capacity=100_000, seed=0):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity, self.rows, self.position = capacity, [], 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, reward, next_observation, terminated, truncated, episode_id, env_t):
        keys = set(observation)
        if any(set(v) != keys for v in (action, reward, next_observation, terminated, truncated)):
            raise ValueError("Joint transition must include every fixed agent")
        row = copy.deepcopy(
            (observation, action, reward, next_observation, terminated, truncated, episode_id, env_t)
        )
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.position] = row
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        if not self.rows:
            raise ValueError("Cannot sample empty replay")
        return [self.rows[i] for i in self.rng.integers(0, len(self.rows), size=size)]

    def get_state(self):
        return copy.deepcopy(
            {
                "capacity": self.capacity,
                "rows": self.rows,
                "position": self.position,
                "rng": self.rng.bit_generator.state,
            }
        )

    def set_state(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity mismatch")
        self.rows, self.position = copy.deepcopy(state["rows"]), state["position"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def __len__(self):
        return len(self.rows)
