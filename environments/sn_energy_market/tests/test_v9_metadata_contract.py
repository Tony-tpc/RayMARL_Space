"""Metadata contracts: configurable tariff calendar and stable FIFO task identity."""

from dataclasses import replace
from datetime import datetime

import pytest

from snmarket import ConsumerState, EnvironmentConfig, ShiftTask
from snmarket.contracts import ConsumerAction, UCAction
from snmarket.tariff import uc_price


@pytest.mark.parametrize(("clock", "expected"), [
    ("08:59", 0.4), ("09:00", 0.6), ("14:59", 0.6), ("15:00", 0.4),
    ("19:59", 0.4), ("20:00", 0.7), ("21:59", 0.7), ("22:00", 0.4),
])
def test_default_tariff_calendar_is_unchanged(clock, expected):
    timestamp = datetime.fromisoformat(f"2026-01-01T{clock}:00+08:00")
    assert uc_price(timestamp, 0, EnvironmentConfig()) == expected


def test_configured_half_open_windows_and_monthly_tiers():
    base = EnvironmentConfig()
    market = replace(base.market, uc_high_intervals_hours=((7.5, 9.0), (13.0, 14.0)),
                     uc_peak_intervals_hours=((18.25, 21.0),))
    config = replace(base, market=market).validate()
    for clock, expected in (("07:29", .5), ("07:30", .7), ("09:00", .5),
                            ("13:00", .7), ("14:00", .5), ("18:15", .8), ("21:00", .5)):
        timestamp = datetime.fromisoformat(f"2026-01-01T{clock}:00+08:00")
        assert uc_price(timestamp, 100.0, config) == expected
    all_flat = replace(config, market=replace(market, uc_high_intervals_hours=(),
                                              uc_peak_intervals_hours=())).validate()
    assert uc_price(datetime.fromisoformat("2026-01-01T20:00:00+08:00"), 0, all_flat) == .4


@pytest.mark.parametrize(("high", "peak"), [
    (((9.0, 9.0),), ()), (((-1.0, 1.0),), ()), (((23.0, 25.0),), ()),
    (((9.0, 12.0), (11.0, 13.0)), ()), (((9.0, 15.0),), ((14.0, 16.0),)),
    (((float("nan"), 15.0),), ()), (((True, 15.0),), ()),
])
def test_invalid_and_overlapping_tariff_windows_are_rejected(high, peak):
    config = EnvironmentConfig()
    with pytest.raises(ValueError):
        replace(config, market=replace(config.market, uc_high_intervals_hours=high,
                                       uc_peak_intervals_hours=peak)).validate()


def test_adjacent_tariff_windows_and_nested_serialization():
    config = EnvironmentConfig.from_dict({"market": {
        "uc_high_intervals_hours": [[9.0, 15.0]], "uc_peak_intervals_hours": [[15.0, 18.0]],
    }})
    assert config.market.uc_high_intervals_hours == ((9.0, 15.0),)
    assert uc_price(datetime.fromisoformat("2026-01-01T15:00:00+08:00"), 0, config) == .7
    assert EnvironmentConfig.from_dict(config.to_dict()) == config


def test_initial_task_ids_are_deterministic_and_do_not_reorder_tasks(make_engine):
    supplied = (
        ShiftTask(-3, 1, .5),
        ShiftTask(-2, 2, .5, "external-request"),
        ShiftTask(-3, 1, .25),
    )
    initial = ConsumerState("consumer_0", tasks=supplied)
    engine = make_engine([[4.0]] * 6, flex=4, rigid=0, deadline=4)
    engine.reset(seed=1, options={"consumer_states": (initial,)})
    tasks = engine.state.consumers[0].tasks
    assert [t.task_id for t in tasks] == [
        "consumer_0:initial:-3:0", "external-request", "consumer_0:initial:-3:2",
    ]
    assert [(t.source_period, t.deadline_period, t.remaining_kwh) for t in tasks] == [
        (t.source_period, t.deadline_period, t.remaining_kwh) for t in supplied
    ]
    engine.reset(seed=999, options={"consumer_states": (initial,)})
    assert engine.state.consumers[0].tasks == tasks


@pytest.mark.parametrize("invalid_id", ["", " ", 7])
def test_initial_explicit_invalid_ids_are_rejected(make_engine, invalid_id):
    engine = make_engine([[4.0]] * 6, flex=4, rigid=0)
    state = ConsumerState("consumer_0", tasks=(ShiftTask(-1, 3, .25, invalid_id),))
    with pytest.raises(ValueError, match="Task IDs"):
        engine.reset(options={"consumer_states": (state,)})


def test_duplicate_initial_ids_and_missing_snapshot_id_are_rejected(make_engine):
    engine = make_engine([[4.0]] * 6, flex=4, rigid=0)
    duplicate = ConsumerState("consumer_0", tasks=(
        ShiftTask(-2, 2, .25, "same"), ShiftTask(-1, 3, .25, "same"),
    ))
    with pytest.raises(ValueError, match="Task IDs"):
        engine.reset(options={"consumer_states": (duplicate,)})
    engine.reset(options={"consumer_states": (
        ConsumerState("consumer_0", tasks=(ShiftTask(-1, 3, .25),)),
    )})
    snapshot = engine.get_state()
    consumer = snapshot.state.consumers[0]
    corrupted = replace(consumer, tasks=(replace(consumer.tasks[0], task_id=None),))
    with pytest.raises(ValueError, match="Task IDs"):
        engine.set_state(replace(snapshot, state=replace(snapshot.state, consumers=(corrupted,))))


def test_new_task_identity_survives_partial_repayment_and_replay(make_engine):
    engine = make_engine([[4.0]] * 6, flex=4, rigid=0, deadline=2)
    engine.reset(seed=1)
    uc = UCAction(grid_import_offer_kw=6, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0)
    first = engine.run_period(uc, {"consumer_0": ConsumerAction(curtail_kw=0, signed_shift_kw=1, der_preference_fraction=0, der_bid_per_kwh=.3)})
    original = first.services["consumer_0"].tasks_after[0]
    assert original.task_id == "consumer_0:shift:0"
    assert original.remaining_kwh == .25
    snapshot = engine.get_state()
    second = engine.run_period(uc, {"consumer_0": ConsumerAction(curtail_kw=0, signed_shift_kw=-.5, der_preference_fraction=0, der_bid_per_kwh=.3)})
    remaining = second.services["consumer_0"].tasks_after[0]
    assert remaining.task_id == original.task_id
    assert remaining.remaining_kwh == .125
    engine.set_state(snapshot)
    replay = engine.run_period(uc, {"consumer_0": ConsumerAction(curtail_kw=0, signed_shift_kw=-.5, der_preference_fraction=0, der_bid_per_kwh=.3)})
    assert replay == second


def test_external_initial_identifier_cannot_collide_with_a_new_task(make_engine):
    engine = make_engine([[4.0]] * 6, flex=4, rigid=0, deadline=4)
    initial = ConsumerState("consumer_0", tasks=(ShiftTask(-1, 3, .25, "consumer_0:shift:0"),))
    engine.reset(options={"consumer_states": (initial,)})
    record = engine.run_period(UCAction(grid_import_offer_kw=6, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0),
                               {"consumer_0": ConsumerAction(curtail_kw=0, signed_shift_kw=1, der_preference_fraction=0, der_bid_per_kwh=.3)})
    tasks = record.services["consumer_0"].tasks_after
    assert [task.task_id for task in tasks] == ["consumer_0:shift:0", "consumer_0:shift:0:1"]
    assert sum(t.remaining_kwh for t in tasks) == .5

