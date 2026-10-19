"""Independent diagnostic calculations and passive optimizer observation."""

import copy
from types import SimpleNamespace

import numpy as np
import torch

from experiments.sn_energy_market.pilot import (
    TrainingAudit,
    assert_same_state,
    parameter_changes,
    reference_advantages,
)


def test_reference_advantages_stops_at_true_terminal_and_bootstraps_slice():
    rows = {
        "rewards": np.array([1.0, 2.0, 3.0]),
        "vf_preds": np.array([0.4, 0.5, 0.6]),
        "next_vf_preds": np.array([0.5, 0.0, 0.7]),
        "terminateds": np.array([False, True, False]),
    }
    np.testing.assert_allclose(reference_advantages(rows, 0.9, 0.8), [2.13, 1.5, 3.03])


def test_parameter_changes_separates_actor_and_value():
    before = {"uc": {"actor.weight": np.zeros(2), "value.weight": np.zeros(1)}}
    after = {
        "uc": {"actor.weight": np.array([3.0, 4.0]), "value.weight": np.array([2.0])}
    }
    assert parameter_changes(before, after) == {"uc": {"actor": 5.0, "value": 2.0}}


def test_optimizer_hooks_ignore_frozen_steps_and_preserve_adam_state(tmp_path):
    modules = {
        name: SimpleNamespace(actor=torch.nn.Linear(2, 1), value=torch.nn.Linear(2, 1))
        for name in ("uc", "consumer")
    }
    optimizers = {
        f"{name}/{role}": torch.optim.Adam(getattr(module, role).parameters(), lr=1e-3)
        for name, module in modules.items()
        for role in ("actor", "value")
    }
    learner = SimpleNamespace(
        module=modules,
        _named_optimizers=optimizers,
        _optimizer_name_to_module={key: key.split("/")[0] for key in optimizers},
        config=SimpleNamespace(grad_clip=0.5),
    )
    audit = TrainingAudit(None, tmp_path)
    audit.install_optimizer_hooks(learner)

    def update(selected):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        for name in selected:
            for role in ("actor", "value"):
                network = getattr(modules[name], role)
                network(torch.ones(3, 2)).sum().backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        for optimizer in optimizers.values():
            optimizer.step()

    try:
        update(("uc", "consumer"))
        frozen = {
            key: copy.deepcopy(opt.state_dict())
            for key, opt in optimizers.items()
            if key.startswith("consumer/")
        }
        update(("uc",))
        assert len(audit.optimizer_steps) == 6
        assert audit.coverage["frozen_optimizer_checks"] == 2
        for key, state in frozen.items():
            assert_same_state(state, optimizers[key].state_dict())
        assert all(
            row["raw_grad_norm"] > 0 and row["parameter_delta"] > 0
            for row in audit.optimizer_steps
        )
        assert all(
            row["clipped_grad_norm"] <= 0.500001 for row in audit.optimizer_steps
        )
    finally:
        audit.close()
