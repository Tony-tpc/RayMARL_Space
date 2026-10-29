"""Bounded UC value-only replay; PPO Actor batches remain on-policy."""

import copy

import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch


class CriticReplay:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.entries = []
        self.rng = np.random.default_rng(seed)

    def mix(self, data):
        if self.capacity == 0:
            return data
        self.entries.append({key: np.array(value, copy=True) for key, value in data.items()})
        self.entries = self.entries[-self.capacity:]
        # Identity on the first phase, without consuming replay or global RNG.
        if len(self.entries) == 1:
            return SampleBatch(copy.deepcopy(self.entries[0]))
        pool = {key: np.concatenate([entry[key] for entry in self.entries]) for key in data}
        indices = self.rng.choice(len(pool["obs"]), len(data["obs"]), replace=False)
        return SampleBatch({key: value[indices].copy() for key, value in pool.items()})

    def get_state(self):
        return copy.deepcopy({"entries": self.entries, "rng": self.rng.bit_generator.state})

    def set_state(self, state):
        self.entries = copy.deepcopy(state["entries"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
