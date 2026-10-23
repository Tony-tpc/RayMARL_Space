"""Structural sampling contract; no simulator or experiment imports."""

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np


class Announcement(Protocol):
    uc_observation: np.ndarray
    uc_action: np.ndarray
    mapping_context: Mapping[str, float]
    consumer_observations: Mapping[str, np.ndarray]
    consumer_bounds: Mapping[str, np.ndarray]
    consumer_value_state: np.ndarray
    period_id: int


class PeriodTransition(Protocol):
    rewards: Mapping[str, float]
    terminated: bool
    next_uc_observation: np.ndarray | None
    metrics: Mapping[str, float]
    data_exhausted: bool


class TrainingEnvironment(Protocol):
    """A factory provides this interface for any compatible two-stage scenario.

    The adapter owns physical feasibility and state. The runner owns policy
    samples, per-period advantages and decision-time value inputs.
    """

    consumer_ids: tuple[str, ...]
    uc_observation_dim: int
    uc_soc_input: Mapping[str, float | int]
    consumer_observation_dim: int
    consumer_value_dim: int
    episode_mode: str

    def reset(self, seed: int) -> np.ndarray: ...
    def uc_bounds(self) -> np.ndarray: ...
    def publish(self, raw_controls: np.ndarray) -> Announcement: ...
    def step(self, consumer_actions: Mapping[str, np.ndarray]) -> PeriodTransition: ...
    def get_state(self) -> Any: ...
    def set_state(self, snapshot: Any) -> None: ...
    def append_profiles(self, profiles: Any) -> np.ndarray: ...


class ScenarioFactory(Protocol):
    def __call__(self, seed: int) -> TrainingEnvironment: ...
    def get_contract(self) -> Mapping[str, Any]: ...
