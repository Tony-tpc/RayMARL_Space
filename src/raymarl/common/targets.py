"""Target updates and finite-state validation."""

import math

import numpy as np
import torch


def soft_update(target, online, tau):
    if not math.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError("tau must be finite and in (0,1]")
    with torch.no_grad():
        for t, s in zip(target.parameters(), online.parameters(), strict=True):
            t.lerp_(s, tau)


def soft_target(rewards, terminated, next_q, next_logp, gamma, temperature):
    return rewards + gamma * (1 - terminated) * (next_q - temperature * next_logp)


def score_function_loss(logp, q, baseline, temperature):
    return (logp * (temperature * logp - q + baseline).detach()).mean()


def validate_finite(state):
    if isinstance(state, dict):
        for value in state.values():
            validate_finite(value)
    elif isinstance(state, (tuple, list)):
        for value in state:
            validate_finite(value)
    elif isinstance(state, torch.Tensor) and not torch.isfinite(state).all():
        raise ValueError("Checkpoint contains non-finite tensors")
    elif (
        isinstance(state, np.ndarray)
        and np.issubdtype(state.dtype, np.number)
        and not np.isfinite(state).all()
    ):
        raise ValueError("Checkpoint contains non-finite arrays")
