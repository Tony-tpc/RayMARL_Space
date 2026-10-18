"""Decision inputs and data provenance remain reconstructible outside actor feedback."""
from datetime import timedelta

import numpy as np
import pytest

from snmarket import Profiles


def test_record_keeps_complete_decision_inputs_and_capacity_decomposition(make_engine):
    engine = make_engine([[4], [4]], der=5, flex=2)
    observed_uc = engine.reset()
    before = engine.state
    observed_consumers = engine.apply_uc_action([8, 1, 1, 1, 0, 2])
    bounds = engine.local_bounds()
    record = engine.apply_consumer_actions({"consumer_0": [0, 0, .5, .3]})
    assert record.decision_state == before
    np.testing.assert_array_equal(record.uc_observation, observed_uc)
    for identity in engine.consumer_ids:
        np.testing.assert_array_equal(record.consumer_observations[identity], observed_consumers[identity])
    assert record.local_bounds == bounds
    diagnosis = record.supply_diagnostics
    reductions = sum(v for key, v in diagnosis.items() if key != "physical_max_supply_kw")
    assert diagnosis["physical_max_supply_kw"] - reductions == pytest.approx(record.ledger.announced_capacity_kw)
    record.consumer_observations.clear()
    record.local_bounds.clear()
    record.supply_diagnostics.clear()
    assert engine.latest_record.consumer_observations
    assert engine.latest_record.local_bounds
    assert engine.latest_record.supply_diagnostics


def test_profile_metadata_is_detached_bound_and_preserved_on_append():
    def block(start, metadata):
        return Profiles([[4]], [1], [.5], [[1]], start_datetime=start,
                        resolution_minutes=15, metadata=metadata)
    start = "2026-01-01T00:00:00+08:00"
    first = block(start, {"source": "sample-a", "value_kind": "realized", "seed": 7, "split": "validation"})
    detached = first.metadata
    detached["seed"] = 999
    assert first.metadata["seed"] == 7
    changed = block(start, {"source": "sample-b"})
    assert first.fingerprint() != changed.fingerprint()
    second = block(first.timestamp(1).isoformat(), {"source": "sample-c"})
    joined = first.append(second)
    assert len(joined) == 2
    assert [s["metadata"]["source"] for s in joined.metadata["segments"]] == ["sample-a", "sample-c"]
    third = block((second.start_datetime + timedelta(minutes=15)).isoformat(), {"source": "sample-d"})
    assert len(joined.append(third).metadata["segments"]) == 3


def test_append_cannot_switch_local_calendar_offset():
    first = Profiles([[4]], [1], [.5], [[1]], start_datetime="2026-01-01T00:00:00+08:00", resolution_minutes=15)
    second = Profiles([[4]], [1], [.5], [[1]], start_datetime="2025-12-31T16:15:00+00:00", resolution_minutes=15)
    with pytest.raises(ValueError, match="contiguous"):
        first.append(second)
