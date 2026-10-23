"""Native continuation/configuration and passive audit equivalence on a tiny environment."""

import copy

import pytest
import torch

from experiments.sn_energy_market.continue_storage_study import change_candidate_seeds
from experiments.sn_energy_market.training_audit import TrainingAudit, assert_same_state
from RLAlgo.tests.conftest import ray_runtime  # noqa: F401
from RLAlgo.tests.test_stackelberg_algorithm import configuration


@pytest.mark.integration
def test_light_audit_and_candidate_protocol_preserve_native_state(ray_runtime, tmp_path):  # noqa: F811
    torch.set_num_threads(1)
    config = configuration(tmp_path).training(deviation_updates=0)
    config.validation_seeds = (11, 12, 13, 14)
    algorithm = config.build_algo()
    try:
        initial = copy.deepcopy(algorithm.get_state())
        audited = []
        for full in (True, False):
            algorithm.set_state(initial)
            audit = TrainingAudit(algorithm, tmp_path, full_period_checks=full).install()
            try:
                algorithm.train()
                audited.append((copy.deepcopy(algorithm.learner_group.get_state()),
                                copy.deepcopy(algorithm.get_state()["raymarl_rng"])))
                assert audit.coverage["refreshed_gae_checks"] > 0
                assert audit.optimizer_steps and audit.coverage["frozen_optimizer_checks"] > 0
                if not full:
                    assert audit.coverage["oracle_comparisons"] == 0
            finally:
                audit.close()
        assert_same_state(audited[0], audited[1])
        before = algorithm.get_state()
        change_candidate_seeds(algorithm, (11, 13))
        after = algorithm.get_state()
        assert after["learning_contract"]["validation_seeds"] == [11, 13]
        for key in before:
            if key != "learning_contract":
                assert_same_state(before[key], after[key])
        assert algorithm.env_runner.config.validation_seeds == (11, 13)
        assert algorithm.learner_group._learner.config.validation_seeds == (11, 13)
        checkpoint = algorithm.save_to_path(str(tmp_path / "two-month-checkpoint"))
        expected = algorithm.env_runner.collect(4)
        algorithm.restore_from_path(checkpoint)
        actual = algorithm.env_runner.collect(4)
        for key, rows in expected.policy_batches.items():
            assert_same_state(dict(rows), dict(actual[key]))
        algorithm.train()
        for candidate in algorithm.last_round["candidates"]:
            assert [row["seed"] for row in candidate["rows"]] == [11, 13]
        with pytest.raises(AssertionError):
            change_candidate_seeds(algorithm, (11,))
    finally:
        algorithm.stop()
