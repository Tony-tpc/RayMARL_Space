"""Reject obsolete v5 entry points instead of reinterpreting equal-sized actions."""
import ast
import importlib
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

import snmarket
from snmarket import ConsumerAction, EnvironmentConfig, UCAction, contracts


@pytest.mark.parametrize("module", [snmarket, contracts])
@pytest.mark.parametrize("name", ["UCActionV9", "ConsumerActionV9"])
def test_single_action_api_has_no_parallel_version_names(module, name):
    assert not hasattr(module, name)


@pytest.mark.parametrize("action, values", [(UCAction, [0.0] * 6), (ConsumerAction, [0, 0, .5, .3])])
def test_actions_require_field_names_to_prevent_positional_reinterpretation(action, values):
    with pytest.raises(TypeError):
        action(*values)


@pytest.mark.parametrize("values", [[0.0] * 5, [0.0] * 7])
def test_v5_uc_vector_shapes_fail_without_mutation(make_engine, values):
    engine = make_engine([[4.0], [4.0]], flex=2)
    engine.reset()
    before = engine.get_state()
    with pytest.raises(ValueError, match="shape"):
        engine.apply_uc_action(values)
    assert engine.get_state() == before


def test_v5_consumer_fields_and_object_are_rejected(make_engine):
    @dataclass(frozen=True)
    class V5ConsumerAction:
        dr_reduce_kw: float
        dr_shift_kw: float
        uc_bid_price: float
        der_bid_price: float

    old = V5ConsumerAction(0.0, 0.0, 0.5, 0.3)
    with pytest.raises(TypeError):
        ConsumerAction(**vars(old))
    engine = make_engine([[4.0], [4.0]], flex=2)
    engine.reset()
    engine.apply_uc_action(UCAction(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0))
    before = engine.get_state()
    with pytest.raises((TypeError, ValueError)):
        engine.apply_consumer_actions({"consumer_0": old})
    assert engine.get_state() == before
    assert [f.name for f in fields(ConsumerAction)] == [
        "curtail_kw", "signed_shift_kw", "der_preference_fraction", "der_bid_per_kwh"]
    # Bare vectors have the documented v9 physical meaning; their origin cannot
    # be inferred from four numbers. This value means rho=.5, never a UC bid.
    record = engine.apply_consumer_actions({"consumer_0": [0, 0, .5, .3]})
    assert record.consumer_actions["consumer_0"].der_preference_fraction == .5
    assert record.orders[0].uc_tariff == .4


@pytest.mark.parametrize("group, name", [
    ("system", "ess_decay_per_period"), ("system", "uc"), ("system", "consumers"),
    ("market", "uc_bid_cap"), ("market", "backup_capacity_limit_kw"),
    ("market", "backup_external_service_rate_per_kwh"), ("market", "mechanism"),
    ("market", "tie_breaking"), ("market", "backup_source"),
    ("rewards", "backup_activation_rate_per_kwh"), ("rewards", "uc_native_loss_rate_per_kwh"),
    ("rewards", "uc_overdue_rate_per_kwh_hour"),
])
def test_legacy_configuration_keys_fail_even_with_current_version(group, name):
    values = EnvironmentConfig().to_dict()
    values[group][name] = 0
    with pytest.raises(ValueError, match="legacy"):
        EnvironmentConfig.from_dict(values)


def test_old_snapshot_signature_rejected_at_both_stages(make_engine):
    engine = make_engine([[4.0], [4.0]])
    engine.reset()
    for action in (None, UCAction(grid_import_offer_kw=4, der_planned_curtailment_kw=0, der_to_ess_kw=0, grid_to_ess_kw=0, ess_discharge_offer_kw=0, dr_procurement_cap_kw=0)):
        if action is not None:
            engine.apply_uc_action(action)
        before = engine.get_state()
        old = replace(before, compatibility_signature="v5_Q1A_Q2A_Q3B_Q4A_Q5B")
        with pytest.raises(ValueError, match="version"):
            engine.set_state(old)
        assert engine.get_state() == before


@pytest.mark.parametrize("kind, name", [
    (contracts.PublicOffer, "der_available_kw"), (contracts.PublicOffer, "uc_available_kw"),
    (contracts.PhysicalExecution, "der_curtail_kw"),
])
def test_obsolete_result_aliases_removed(kind, name):
    assert not hasattr(kind, name)


def test_old_decoder_and_dispatch_functions_are_not_exported():
    for module_name, names in {
        "action": ("decode_uc", "decode_consumer", "normalized_action"),
        "physics": ("prepare_regular_dispatch", "finalize_execution"),
    }.items():
        module = importlib.import_module(f"snmarket.{module_name}")
        for name in names:
            assert not hasattr(module, name)


def test_reserve_module_and_old_contracts_are_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("snmarket.reserve")
    for name in ("RegularDispatch", "Fulfillment", "DRRequest"):
        assert not hasattr(contracts, name)
    source = Path(snmarket.__file__).parent
    for path in source.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(isinstance(node, ast.ImportFrom) and node.module in ("reserve", "snmarket.reserve")
                       for node in ast.walk(tree)), path.name


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
