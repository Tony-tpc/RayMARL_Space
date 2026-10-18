"""A matching instant must not silently change the model's local tariff calendar."""
from dataclasses import replace
from datetime import timezone

import pytest

from snmarket import EnergyMarketEngine, EnvironmentConfig, Profiles


def test_same_instant_different_local_offset_rejected_at_construction():
    base = EnvironmentConfig()
    config = replace(base, system=replace(base.system, num_consumers=1, horizon_steps=1,
                                        start_datetime="2026-01-01T09:00:00+08:00"))
    profiles = Profiles([[4]], [0], [.65], [[0]], start_datetime="2026-01-01T01:00:00+00:00",
                        resolution_minutes=15)
    with pytest.raises(ValueError, match="calendar"):
        EnergyMarketEngine(config, profiles)


def test_snapshot_cannot_change_local_tariff_calendar(make_engine):
    engine = make_engine([[4], [4]], start="2026-01-01T09:00:00+08:00")
    engine.reset()
    snapshot = engine.get_state()
    changed = replace(snapshot, state=replace(snapshot.state,
                       timestamp=snapshot.state.timestamp.astimezone(timezone.utc)))
    with pytest.raises(ValueError, match="calendar"):
        engine.set_state(changed)
    assert engine.get_state() == snapshot


@pytest.mark.parametrize("identity", ["", 0, False])
def test_explicit_invalid_transition_id_does_not_use_default(make_engine, identity):
    engine = make_engine([4])
    engine.reset()
    engine.apply_uc_action([4, 0, 0, 0, 0, 0])
    snapshot = engine.get_state()
    with pytest.raises(ValueError, match="Transition ID"):
        engine.apply_consumer_actions({"consumer_0": [0, 0, 0, .3]}, transition_id=identity)
    assert engine.get_state() == snapshot
