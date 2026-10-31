"""Execution parity for persistent lanes, temporary announcements and paired evaluation."""

import copy
import pickle
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import torch
from experiments.sn_energy_market.training_audit import assert_same_state
from RLAlgo.tests.test_stackelberg_algorithm import configuration
from RLAlgo.tests.test_stackelberg_sampling import assert_batches_equal

from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner, random_state, restore_random_state


def mixed_config(path, workers=0, device="cpu"):
    cfg = configuration(path, device=device).training(phase_sampling="all_streams", deviation_updates=0,
                                       execution_workers=workers)
    cfg.train_seeds = (2, 3, 4, 5)
    cfg.train_seed_groups = tuple((s,) for s in cfg.train_seeds)
    cfg.prepare_policies()
    return cfg


def test_temporary_bootstrap_matches_full_snapshot_and_restores_on_failure(tmp_path):
    torch.set_num_threads(1)
    runner = StackelbergEnvRunner(config=mixed_config(tmp_path))
    try:
        runner.collect(2)
        before = runner.get_state()
        history = runner.env.engine._records

        @contextmanager
        def full_snapshot():
            snapshot = runner.env.get_state()
            try:
                yield
            finally:
                runner.env.set_state(snapshot)

        with patch.object(runner, "_temporary_announcement", full_snapshot):
            expected = runner.bootstrap_inputs()
        runner.set_state(before)
        history = runner.env.engine._records
        with patch.object(runner.env, "get_state", side_effect=AssertionError("history copy")):
            actual = runner.bootstrap_inputs()
        assert_same_state(expected, actual)
        assert runner.env.engine._records is history
        assert_same_state(before, runner.get_state())
        publish = runner.env.publish

        def fail_after_publish(*args):
            publish(*args)
            raise RuntimeError("injected announcement failure")

        with (patch.object(runner.env, "publish", fail_after_publish),
              pytest.raises(RuntimeError, match="injected")):
            runner.bootstrap_inputs()
        assert_same_state(before, runner.get_state())
    finally:
        runner.stop()


def test_resident_batches_rng_branches_checkpoint_and_all_candidate_months(tmp_path):
    torch.set_num_threads(1)
    serial = StackelbergEnvRunner(config=mixed_config(tmp_path / "serial"))
    cfg = mixed_config(tmp_path / "parallel", 4)
    parallel = StackelbergEnvRunner(config=cfg)
    recovered = None
    try:
        saved = serial.get_state()
        parallel.set_state(saved)
        rng = random_state()
        reference = serial.collect_phase(16)
        reference_state = serial.get_state()
        restore_random_state(rng)
        actual = parallel.collect_phase(16)
        assert_batches_equal(reference, actual)
        assert_same_state(reference_state, parallel.get_state())
        branch = parallel.branch_state()
        serial_branch = serial.get_state()
        reference = serial.collect_phase(32)  # Cross true month boundaries in all four streams.
        reference_state = serial.get_state()
        with patch.object(parallel._execution, "export", side_effect=AssertionError("hot history transfer")):
            actual = parallel.collect_phase(32)
        assert_batches_equal(reference, actual)
        assert_same_state(reference_state, parallel.get_state())
        parallel.set_state(branch)
        serial.set_state(serial_branch)
        assert_same_state(serial.get_state(), parallel.get_state())
        assert_batches_equal(serial.collect_phase(8), parallel.collect_phase(8))

        checkpoint = pickle.loads(pickle.dumps(parallel.get_state()))
        policies = [copy.deepcopy(parallel.module.get_state()) for _ in range(2)]
        for module in policies[1].values():
            for key, value in module.items():
                if "actor" in key:
                    module[key] = value + 0.01
        serial.set_state(checkpoint)
        expected = serial.evaluate_candidates(policies, (11, 12))
        observed = parallel.evaluate_candidates(policies, (11, 12))
        assert_same_state(expected, observed)
        assert len(observed) == 2 and all(len(rows) == 2 for rows in observed)
        assert all(row["physical_periods"] == 6 for rows in observed for row in rows)
        assert_same_state(checkpoint, parallel.get_state())
        parallel.release_branches()
        expected = parallel.collect_phase(16)
        processes = parallel._execution.processes.copy()
        parallel.stop()
        assert not any(process.is_alive() for process in processes)
        recovered = StackelbergEnvRunner(config=cfg)
        recovered.set_state(checkpoint)
        assert_batches_equal(expected, recovered.collect_phase(16))
        # A dead child must surface immediately, not leave the controller waiting on a pipe.
        recovered._execution.processes[0].terminate()
        recovered._execution.processes[0].join(timeout=5)
        with pytest.raises(RuntimeError, match="exited|disconnected"):
            recovered._execution.receive(0)
    finally:
        serial.stop()
        parallel.stop()
        if recovered is not None:
            recovered.stop()


def test_execution_settings_do_not_change_learning_contract(tmp_path):
    serial = mixed_config(tmp_path)
    parallel = mixed_config(tmp_path, 4)
    assert serial.learning_contract() == parallel.learning_contract()
    parallel.validate()
    parallel.execution_workers = 3
    with pytest.raises(ValueError, match="one worker per"):
        parallel.validate()


def test_conditional_returns_match_resident_sampling_preserve_branches_and_ignore_values(tmp_path):
    torch.set_num_threads(1)
    serial = StackelbergEnvRunner(config=mixed_config(tmp_path / "serial"))
    parallel = StackelbergEnvRunner(config=mixed_config(tmp_path / "parallel", 4))
    try:
        parallel.set_state(serial.get_state())
        reference = serial.collect_phase(16)
        assert_batches_equal(reference, parallel.collect_phase(16))
        before, rng = serial.get_state(), random_state()
        branch = parallel.branch_state()
        options = {"stride": 2, "horizon": 4, "replicates": 2, "seed": 313}
        expected, cost = serial.collect_mode_returns(16, **options)
        assert_same_state(before, serial.get_state())
        assert_same_state(rng, random_state())
        actual, actual_cost = parallel.collect_mode_returns(16, **options)
        assert_same_state(dict(expected), dict(actual))
        assert cost == actual_cost and 16 < cost <= 16 + 8 * 2 * 3 * 4
        assert len(actual["obs"]) == 8
        assert_same_state(before, parallel.get_state())
        # The earlier native candidate snapshot must still be usable.
        parallel.set_state(branch)
        assert_same_state(before, parallel.get_state())
        for module in serial.module.values():
            with torch.no_grad():
                for parameter in module.value.parameters():
                    parameter.add_(100.)
        repeated, repeated_cost = serial.collect_mode_returns(16, **options)
        assert_same_state(dict(expected), dict(repeated))
        assert repeated_cost == cost
        for mode in range(3):
            assert np.isfinite(actual["mode_q_values"][:, mode]).all()
    finally:
        serial.stop()
        parallel.stop()


def test_failed_conditional_probe_restores_training_cursor_and_rng(tmp_path):
    torch.set_num_threads(1)
    runner = StackelbergEnvRunner(config=mixed_config(tmp_path))
    try:
        runner.collect_phase(8)
        before, rng = runner.get_state(), random_state()
        with (
            patch.object(runner.module["uc"], "draw_conditional_mode", side_effect=RuntimeError("probe failed")),
            pytest.raises(RuntimeError, match="probe failed"),
        ):
            runner.collect_mode_returns(8, stride=1, horizon=2, replicates=1, seed=9)
        assert_same_state(before, runner.get_state())
        assert_same_state(rng, random_state())
    finally:
        runner.stop()


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resident_round_matches_serial_weights_adam_rng_and_budget(ray_runtime, tmp_path, device):
    from experiments.sn_energy_market.training_trace import TrainingTrace

    torch.set_num_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    serial_cfg = mixed_config(tmp_path / "serial", device=device)
    serial = serial_cfg.build_algo()
    parallel_cfg = mixed_config(tmp_path / "parallel", 4, device=device)
    parallel = parallel_cfg.build_algo()
    audit = None
    try:
        initial = serial.get_state()
        parallel.set_state(initial)
        serial.train()
        expected = serial.get_state()
        parallel.set_state(initial)
        audit = TrainingTrace(parallel, tmp_path, write_trajectory=False).install()
        parallel.train()
        audit.assert_budget(1)
        actual = parallel.get_state()
        for key in ("learner_group", "env_runner", "raymarl_rng", "stackelberg_runtime"):
            assert_same_state(expected[key], actual[key])
        assert parallel.completed_updates == 4
        assert parallel.sampled_periods == 16
        assert parallel.evaluated_periods == 24
    finally:
        if audit is not None:
            audit.close()
        serial.stop()
        parallel.stop()
