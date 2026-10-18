"""Space layouts; no environment-specific names or parameters."""

import itertools
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch


@dataclass(frozen=True)
class Branch:
    path: tuple
    kind: str
    size: int
    shape: tuple
    start: int
    low: np.ndarray | None = None
    high: np.ndarray | None = None


class ActionLayout:
    """Lossless layout for bounded Box, Discrete, MultiBinary, Dict and Tuple."""

    def __init__(self, space):
        self.space = space
        self.branches = []

        def visit(s, path=()):
            if isinstance(s, gym.spaces.Dict):
                for key, child in s.spaces.items():
                    visit(child, path + (key,))
            elif isinstance(s, gym.spaces.Tuple):
                for key, child in enumerate(s.spaces):
                    visit(child, path + (key,))
            elif isinstance(s, gym.spaces.Box):
                if not np.isfinite(s.low).all() or not np.isfinite(s.high).all():
                    raise ValueError("v0.1 requires finite Box action bounds")
                if not (s.high > s.low).all():
                    raise ValueError("Box action bounds must have positive width")
                self.branches.append(
                    Branch(
                        path, "box", int(np.prod(s.shape)), s.shape, 0, s.low.reshape(-1), s.high.reshape(-1)
                    )
                )
            elif isinstance(s, gym.spaces.Discrete):
                self.branches.append(Branch(path, "categorical", int(s.n), (), int(s.start)))
            elif isinstance(s, gym.spaces.MultiBinary):
                self.branches.append(Branch(path, "binary", int(np.prod(s.shape)), s.shape, 0))
            else:
                raise TypeError(f"Unsupported action space: {s}")

        visit(space)
        self.param_dim = sum(2 * b.size if b.kind == "box" else b.size for b in self.branches)
        self.feature_dim = sum(b.size for b in self.branches)
        self.is_continuous = all(b.kind == "box" for b in self.branches)
        self.discrete_count = int(
            np.prod([2**b.size if b.kind == "binary" else b.size for b in self.branches if b.kind != "box"])
        )

    def unpack(self, values):
        if not self.branches[0].path:
            return values[0]
        out = {} if isinstance(self.space, gym.spaces.Dict) else []
        for branch, value in zip(self.branches, values):
            node = out
            for depth, key in enumerate(branch.path):
                last = depth == len(branch.path) - 1
                if isinstance(node, list):
                    while len(node) <= key:
                        node.append(None)
                    if last:
                        node[key] = value
                    else:
                        if node[key] is None:
                            node[key] = {} if isinstance(branch.path[depth + 1], str) else []
                        node = node[key]
                elif last:
                    node[key] = value
                else:
                    node = node.setdefault(key, {} if isinstance(branch.path[depth + 1], str) else [])

        def tuples(node):
            if isinstance(node, list):
                return tuple(tuples(v) for v in node)
            if isinstance(node, dict):
                return {k: tuples(v) for k, v in node.items()}
            return node

        return tuples(out)

    def values(self, action):
        out = []
        for branch in self.branches:
            value = action
            for key in branch.path:
                value = value[key]
            out.append(value)
        return out

    def encode(self, action, device="cpu"):
        encoded = []
        for b, value in zip(self.branches, self.values(action)):
            x = torch.as_tensor(value, device=device)
            if b.kind == "categorical":
                x = torch.nn.functional.one_hot(x.long() - b.start, b.size).float()
            else:
                shape = x.shape[: -len(b.shape)] if b.shape else x.shape
                x = x.float().reshape(*shape, b.size)
                if b.kind == "box":
                    low = torch.as_tensor(b.low, device=device)
                    high = torch.as_tensor(b.high, device=device)
                    x = 2 * (x - low) / (high - low) - 1
            encoded.append(x)
        return torch.cat(encoded, dim=-1)

    def decode_features(self, features):
        values, offset = [], 0
        for b in self.branches:
            x = features[..., offset : offset + b.size]
            offset += b.size
            if b.kind == "box":
                lo, hi = x.new_tensor(b.low), x.new_tensor(b.high)
                x = lo + (x + 1) * (hi - lo) / 2
                x = x.reshape(*x.shape[:-1], *b.shape)
            elif b.kind == "categorical":
                x = x.argmax(-1) + b.start
            else:
                x = x.long().reshape(*x.shape[:-1], *b.shape)
            values.append(x)
        return self.unpack(values)

    def discrete_choices(self):
        sets = []
        for b in self.branches:
            if b.kind == "categorical":
                sets.append(list(range(b.size)))
            elif b.kind == "binary":
                sets.append(list(itertools.product((0, 1), repeat=b.size)))
        if self.discrete_count > 256:
            raise ValueError("Exact discrete enumeration limited to 256 combinations")
        return list(itertools.product(*sets))


class ObservationLayout:
    """Vector observations, optionally Dict(current, history, history_mask)."""

    def __init__(self, space):
        self.space = space
        self.history_shape = None
        if isinstance(space, gym.spaces.Dict) and "history" in space.spaces:
            self.history_shape = space["history"].shape
            if len(self.history_shape) != 2:
                raise ValueError("history must have shape [lookback, channels]")
            self.current_space = gym.spaces.Dict(
                {k: v for k, v in space.spaces.items() if k not in ("history", "history_mask")}
            )
        else:
            self.current_space = space
        self.current_dim = gym.spaces.utils.flatdim(self.current_space)

    def flatten(self, observation):
        if self.history_shape:
            current = {k: observation[k] for k in self.current_space.spaces}
            h = np.asarray(observation["history"], np.float32)
            mask = np.asarray(observation.get("history_mask", np.ones(h.shape[0])), np.float32)
        else:
            current, h, mask = observation, None, None
        return np.asarray(gym.spaces.utils.flatten(self.current_space, current), np.float32), h, mask

    def tensor(self, observations, device):
        rows = [self.flatten(o) for o in observations]
        current = torch.as_tensor(np.stack([r[0] for r in rows]), device=device)
        history = mask = None
        if self.history_shape:
            history = torch.as_tensor(np.stack([r[1] for r in rows]), device=device)
            mask = torch.as_tensor(np.stack([r[2] for r in rows]), device=device)
        return {"current": current, "history": history, "history_mask": mask}

    def torch_batch(self, observation):
        def flatten(s, value):
            if isinstance(s, gym.spaces.Dict):
                return torch.cat([flatten(child, value[k]) for k, child in s.spaces.items()], -1)
            if isinstance(s, gym.spaces.Tuple):
                return torch.cat([flatten(child, value[i]) for i, child in enumerate(s.spaces)], -1)
            if isinstance(s, gym.spaces.Discrete):
                return torch.nn.functional.one_hot(value.long() - s.start, s.n).float()
            dimensions = len(s.shape)
            leading = value.shape[:-dimensions] if dimensions else value.shape
            return value.float().reshape(*leading, -1)

        if self.history_shape:
            return {
                "current": flatten(self.current_space, observation),
                "history": observation["history"],
                "history_mask": observation.get(
                    "history_mask", torch.ones_like(observation["history"][..., 0])
                ),
            }
        return {"current": flatten(self.current_space, observation), "history": None, "history_mask": None}
