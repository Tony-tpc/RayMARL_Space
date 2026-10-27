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


def test_temporary_announcement_rejects_commit_without_changing_state(make_engine):
    from snmarket import StageError

    engine = make_engine([[4.0], [4.0]])
    engine.reset()
    before = engine.get_state()
    with pytest.raises(StageError, match="cannot commit"), engine.temporary_announcement():
        engine.apply_uc_action([4, 0, 0, 0, 0, 0])
        engine.apply_consumer_actions({"consumer_0": [0, 0, 0, .3]})
    assert engine.get_state() == before


def test_local_branches_share_history_but_isolate_new_commits_and_public_copies(make_engine):
    engine = make_engine([[4.0], [4.0], [4.0]])
    engine.reset()
    uc, consumers = [4, 0, 0, 0, 0, 0], {"consumer_0": [0, 0, 0, .3]}
    engine.run_period(uc, consumers, transition_id="first")
    canonical = engine.get_state()
    branch = engine._branch_state()
    assert branch.records is not engine._records
    assert branch.records["first"] is engine._records["first"]
    public = engine.get_state()
    public.records.clear()
    assert engine.get_state() == canonical
    expected = engine.run_period(uc, consumers, transition_id="second")
    assert len(branch.records) == 1
    engine._restore_branch(branch)
    assert engine.get_state() == canonical
    assert engine.run_period(uc, consumers, transition_id="second") == expected
    assert len(branch.records) == 1
