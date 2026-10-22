"""Pure numerical checks of matched comparisons and joint stopping conditions."""

import copy
import json

import numpy as np
import pytest

from raymarl.algorithms.stackelberg.evaluation import (
    actor_parameter_change,
    paired_gain,
    select_candidate,
    stopping_status,
    summarize_evaluation,
)


def rows(values):
    return [{"seed": i, "returns": {"uc": value}, "score_complete": True} for i, value in enumerate(values)]


def comparison(mean, width=0.02):
    return {"mean": mean, "lower": mean - width, "upper": mean + width, "found_gain": max(0.0, mean)}


def stopping_inputs():
    consumer = {"searched": True, "per_consumer": {"c0": comparison(0.03), "c1": comparison(-0.2)}}
    uc = [comparison(0, 0), comparison(0.04), comparison(-0.1)]
    change = {"per_agent": {"uc": 0.005, "c0": 0.008, "c1": 0.006}, "maximum_rms": 0.008}
    tolerances = {
        "consumer_gain": 0.05,
        "uc_gain": 0.05,
        "policy_change": 0.01,
        "confidence_half_width": 0.025,
    }
    return consumer, uc, change, tolerances


def test_matched_gain_and_selection_preserve_incumbent_and_uncertainty():
    reference = rows([2, 4, 8])
    better = rows([4, 6, 10])
    noisy = rows([100, -50, 20])
    index, comparisons = select_candidate([reference, better, noisy])
    assert index == 1
    assert comparisons[1]["mean"] == comparisons[1]["lower"] == comparisons[1]["upper"] == 2
    assert select_candidate([reference, rows([1, 3, 7])])[0] == 0
    assert select_candidate([rows([0]), rows([100])])[0] == 0
    with pytest.raises(ValueError, match="same ordered"):
        paired_gain(reference, list(reversed(reference)))


def test_actor_change_counts_scalars_and_ignores_value_network():
    before = {
        "uc": {"actor.a": np.zeros(3), "actor.b": np.zeros(1), "value.a": np.zeros(2)},
        "c0": {"actor.a": np.zeros(2)},
    }
    after = {
        "uc": {"actor.a": np.ones(3), "actor.b": np.array([3.0]), "value.a": np.ones(2) * 1e6},
        "c0": {"actor.a": np.ones(2) * 2},
    }
    actual = actor_parameter_change(before, after)
    assert actual["per_agent"]["uc"] == pytest.approx(np.sqrt(3))
    assert actual["maximum_rms"] == actual["per_agent"]["c0"] == 2
    assert actor_parameter_change(after, after)["maximum_rms"] == 0


@pytest.mark.parametrize("change", ["agent", "actor", "shape"])
def test_actor_change_rejects_incomparable_networks(change):
    before = {"uc": {"actor.a": np.zeros(2)}}
    after = copy.deepcopy(before)
    if change == "agent":
        after["c0"] = after["uc"]
    elif change == "actor":
        after["uc"]["actor.b"] = np.zeros(2)
    else:
        after["uc"]["actor.a"] = np.zeros(3)
    with pytest.raises(ValueError, match="matching|same agent"):
        actor_parameter_change(before, after)


def test_joint_stopping_requires_all_four_measures():
    args = stopping_inputs()
    status = stopping_status(*args)
    assert status["passed"] and all(status["criteria"].values())
    assert status["measurements"]["consumer_gain"] == 0.03
    assert status["measurements"]["uc_gain"] == 0.04
    for key in args[-1]:
        reduced = {**args[-1], key: status["measurements"][key] / 2}
        failed = stopping_status(*args[:-1], reduced)
        assert not failed["passed"] and not failed["criteria"][key]


@pytest.mark.parametrize("missing", ["skipped", "empty_consumers", "no_candidate", "consumer_ci", "uc_ci"])
def test_stopping_never_passes_missing_search_or_uncertainty(missing):
    consumer, uc, change, tolerances = stopping_inputs()
    if missing == "skipped":
        consumer["searched"] = False
    elif missing == "empty_consumers":
        consumer["per_consumer"] = {}
    elif missing == "no_candidate":
        uc = uc[:1]
    elif missing == "consumer_ci":
        consumer["per_consumer"]["c0"]["lower"] = None
    else:
        uc[1]["upper"] = None
    status = stopping_status(consumer, uc, change, tolerances)
    assert not status["passed"]
    json.dumps(status, allow_nan=False)


def test_summary_preserves_physical_counts_and_bootstrap_flag():
    values = [
        {
            "returns": {"uc": 2.0, "c0": -1.0},
            "physical_periods": 4,
            "metrics": {"cash": 1.0},
            "bootstrap_used": False,
        },
        {
            "returns": {"uc": 4.0, "c0": -3.0},
            "physical_periods": 7,
            "metrics": {"cash": 5.0},
            "bootstrap_used": True,
        },
    ]
    summary = summarize_evaluation(values)
    assert summary == {
        "scenarios": 2,
        "physical_periods": 11,
        "returns": {"uc": 3.0, "c0": -2.0},
        "metrics": {"cash": 3.0},
        "bootstrap_used": True,
    }
