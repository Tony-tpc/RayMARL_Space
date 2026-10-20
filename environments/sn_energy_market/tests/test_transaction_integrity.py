"""Failed settlement, FIFO or market validation must leave the engine untouched."""
from dataclasses import replace

import pytest


@pytest.mark.parametrize("component", ["cash", "queue", "market"])
def test_commit_checks_still_reject_corrupted_results(make_engine, monkeypatch, component):
    from snmarket import engine as engine_module

    engine = make_engine([[4.0], [4.0]])
    engine.reset()
    engine.apply_uc_action([4, 0, 0, 0, 0, 0])
    before = engine.get_state()
    function = {"cash": "settle", "queue": "realize_service", "market": "clear"}[component]
    original = getattr(engine_module, function)

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        key = {"cash": "cash_residual", "queue": "queue_residual_kwh", "market": "uc_total_kw"}[component]
        return replace(result, **{key: getattr(result, key) + 1})

    monkeypatch.setattr(engine_module, function, corrupt)
    with pytest.raises(ValueError):
        engine.apply_consumer_actions({"consumer_0": [0, 0, 0, .3]})
    assert engine.get_state() == before
