"""Immutable, time-aligned exogenous inputs; no implicit data repetition."""
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


class Profiles:
    def __init__(self, loads_kw, der_kw, grid_cost_per_kwh, flex_kw, *, rigid_kw=None,
                 start_datetime, resolution_minutes, metadata=None):
        information = {"source": "unspecified", "value_kind": "realized", "seed": None, "split": "unspecified"}
        if metadata is not None:
            information.update(dict(metadata))
        if information["value_kind"] not in {"realized", "forecast", "mixed"}:
            raise ValueError("Profile value_kind must be realized, forecast or mixed")
        self._metadata_json = json.dumps(information, sort_keys=True, ensure_ascii=False, allow_nan=False)
        self.start_datetime = datetime.fromisoformat(start_datetime)
        if self.start_datetime.tzinfo is None:
            raise ValueError("Profiles require an explicit UTC offset")
        if isinstance(resolution_minutes, bool) or not isinstance(resolution_minutes, int) or resolution_minutes < 1:
            raise ValueError("Profile resolution must be positive")
        if (self.start_datetime.second or self.start_datetime.microsecond
                or (self.start_datetime.hour * 60 + self.start_datetime.minute) % resolution_minutes):
            raise ValueError("Profile start must align to interval boundaries")
        self.resolution_minutes = resolution_minutes
        if rigid_kw is None:
            rigid_kw = np.maximum(0, np.asarray(loads_kw) - np.asarray(flex_kw))
        for name, values in (("loads_kw", loads_kw), ("der_kw", der_kw),
                             ("grid_cost_per_kwh", grid_cost_per_kwh), ("flex_kw", flex_kw), ("rigid_kw", rigid_kw)):
            arr = np.asarray(values, dtype=np.float64)
            if not np.isfinite(arr).all() or (arr < 0).any():
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, np.frombuffer(arr.tobytes(), dtype=np.float64).reshape(arr.shape))
        if self.loads_kw.ndim != 2 or not all(self.loads_kw.shape):
            raise ValueError("loads_kw must have shape [T,N] with T,N > 0")
        if self.der_kw.shape != (len(self),) or self.grid_cost_per_kwh.shape != (len(self),):
            raise ValueError("DER and price profiles must have shape [T]")
        if self.flex_kw.shape != self.loads_kw.shape or self.rigid_kw.shape != self.loads_kw.shape:
            raise ValueError("flex_kw and rigid_kw must have shape [T,N]")
        if (self.rigid_kw > self.loads_kw).any():
            raise ValueError("Rigid load cannot exceed baseline")
        self._fingerprint = self._compute_fingerprint()
        self._sealed = True

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Profiles are immutable; append a contiguous block through the engine")
        object.__setattr__(self, name, value)

    def __len__(self):
        return self.loads_kw.shape[0]

    @property
    def metadata(self):
        """A detached JSON-compatible provenance record; never an actor observation."""
        return json.loads(self._metadata_json)

    def timestamp(self, period):
        return self.start_datetime + timedelta(minutes=period * self.resolution_minutes)

    def _compute_fingerprint(self):
        digest = hashlib.sha256(f"{self.start_datetime.isoformat()}:{self.resolution_minutes}".encode())
        digest.update(self._metadata_json.encode())
        for arr in (self.loads_kw, self.der_kw, self.grid_cost_per_kwh, self.flex_kw, self.rigid_kw):
            digest.update(str(arr.shape).encode())
            digest.update(arr.tobytes())
        return digest.hexdigest()

    def fingerprint(self):
        return self._fingerprint

    def append(self, other):
        if (not isinstance(other, Profiles) or other.start_datetime != self.timestamp(len(self))
                or other.start_datetime.utcoffset() != self.start_datetime.utcoffset()
                or other.resolution_minutes != self.resolution_minutes
                or other.loads_kw.shape[1] != self.loads_kw.shape[1]):
            raise ValueError("Appended profiles must be contiguous with matching resolution/consumers")
        arrays = {name: np.concatenate((getattr(self, name), getattr(other, name)))
                  for name in ("loads_kw", "der_kw", "grid_cost_per_kwh", "flex_kw", "rigid_kw")}
        def segments(profile):
            meta = profile.metadata
            return meta.get("segments", [{"start_datetime": profile.start_datetime.isoformat(),
                                          "periods": len(profile), "metadata": meta}])
        return Profiles(**arrays, start_datetime=self.start_datetime.isoformat(),
                        resolution_minutes=self.resolution_minutes,
                        metadata={"source": "appended_blocks", "value_kind": self.metadata["value_kind"]
                                  if self.metadata["value_kind"] == other.metadata["value_kind"] else "mixed",
                                  "segments": segments(self) + segments(other)})

def synthetic_profiles(config):
    s = config.system
    start = datetime.fromisoformat(s.start_datetime)
    hour = np.arange(s.horizon_steps) * s.dt_hours + start.hour + start.minute / 60
    ids = np.arange(s.num_consumers)
    loads = 5 + ids[None, :] * 0.7 + 2 * (1 + np.sin(2 * np.pi * (hour[:, None] - 8) / 24))
    der = s.num_consumers * 5 * np.maximum(0, np.sin(np.pi * (hour % 24 - 6) / 12))
    cost = np.where((hour % 24 >= 17) & (hour % 24 < 22), 0.65, 0.35)
    return Profiles(loads, der, cost, loads * s.dr_limit_ratio,
                    start_datetime=s.start_datetime, resolution_minutes=s.resolution_minutes,
                    metadata={"source": "synthetic_sine_v9", "value_kind": "realized"})

def load_profiles(path, config):
    """CSV columns consumer_i_load_kw and optional consumer_i_rigid_kw / consumer_i_flex_kw."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    s = config.system
    if not rows or (s.episode_mode == "finite" and len(rows) != s.horizon_steps):
        raise ValueError("Finite CSV length must equal horizon_steps; continuous CSV must be nonempty")
    loads, flex, rigid, der, prices = [], [], [], [], []
    start = datetime.fromisoformat(s.start_datetime)
    for t, row in enumerate(rows):
        instant = datetime.fromisoformat(row["timestamp"])
        if (instant != start + timedelta(minutes=t * s.resolution_minutes)
                or instant.utcoffset() != start.utcoffset()):
            raise ValueError(f"Missing, duplicated or misaligned timestamp at {t}")
        base = [float(row[f"consumer_{i}_load_kw"]) for i in range(s.num_consumers)]
        ability = [float(row.get(f"consumer_{i}_flex_kw", x * s.dr_limit_ratio)) for i, x in enumerate(base)]
        loads.append(base)
        flex.append(ability)
        rigid.append([float(row.get(f"consumer_{i}_rigid_kw", max(0, x - ability[i])))
                      for i, x in enumerate(base)])
        der.append(float(row["der_kw"]))
        prices.append(float(row["grid_cost_per_kwh"]))
    return Profiles(loads, der, prices, flex, rigid_kw=rigid,
                    start_datetime=s.start_datetime, resolution_minutes=s.resolution_minutes,
                    metadata={"source": str(Path(path).resolve()), "value_kind": "realized",
                              "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})

