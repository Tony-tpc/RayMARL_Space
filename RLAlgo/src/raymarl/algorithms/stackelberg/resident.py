"""Persistent CPU lanes. Full histories cross the pipe only for explicit checkpoints."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from multiprocessing.connection import wait
from uuid import uuid4


@dataclass(frozen=True)
class BranchState:
    owner: str
    token: int
    metadata: dict
    cursors: list
    rng: dict


def _worker(connection, config_blob, lane, cuda_seeds, deterministic):
    # Keep this module free of torch/Ray imports before Windows spawn initialization.
    import random

    import cloudpickle
    import numpy as np
    import torch

    from . import runner as native

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(deterministic)
    cuda_state = None
    original_seed = torch.manual_seed

    def state():
        return {"torch": torch.get_rng_state().clone(), "numpy": np.random.get_state(),
                "python": random.getstate(), "cuda": copy.deepcopy(cuda_state)}

    def restore(saved):
        nonlocal cuda_state
        torch.set_rng_state(torch.as_tensor(saved["torch"], dtype=torch.uint8).cpu())
        np.random.set_state(saved["numpy"])
        random.setstate(saved["python"])
        cuda_state = copy.deepcopy(saved["cuda"])

    def seed(value):
        nonlocal cuda_state
        result = original_seed(value)
        cuda_state = copy.deepcopy(cuda_seeds.get(int(value)))
        return result

    # CUDA is invisible in workers; retain its unconsumed RNG metadata for exact recovery.
    native.random_state, native.restore_random_state = state, restore
    torch.manual_seed = seed
    runner = evaluator = None
    snapshots = {}
    try:
        cfg = cloudpickle.loads(config_blob)
        runner = cfg.env_runner_cls(config=cfg)
        runner.active_lane = lane
        connection.send((True, None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "load":
                saved, lane_steps = payload
                runner.lane_steps = lane_steps.copy()
                runner._load_single(saved)
                snapshots.clear()
                result = runner.lane_summary()
            elif command == "collect":
                weights, steps, count, lane_steps, outside, greedy, observer = payload
                restore(outside)
                runner.trajectory_periods = runner.num_env_steps_sampled_lifetime = count
                runner.lane_steps = lane_steps.copy()
                runner.module.set_state(weights)
                physical, original = [], runner._period
                if observer is not None:
                    observe_before, observe_after = cloudpickle.loads(observer)

                    def period(*, _before=observe_before, _after=observe_after,
                               _original=original, _physical=physical, **kwargs):
                        before = _before(runner.env)
                        samples, transition = _original(**kwargs)
                        _physical.append(_after(runner.env, before, samples, transition))
                        return samples, transition

                    runner._period = period
                try:
                    batch = runner.collect(steps, deterministic=greedy)
                    result = batch, runner._sampling_metadata(), runner.lane_summary(), physical
                finally:
                    runner._period = original
            elif command == "mode_returns":
                weights, steps, options, probe_seed = payload
                runner.module.set_state(weights)
                result = runner._mode_return_lane(steps, **options, seed=probe_seed)
            elif command == "save":
                snapshots[payload] = runner._single_state(branch=True), runner.lane_steps.copy()
                result = None
            elif command == "restore":
                saved, lane_steps = snapshots[payload]
                runner._load_single(saved, branch=True)
                runner.lane_steps = lane_steps.copy()
                result = None
            elif command == "release":
                snapshots.clear()
                result = None
            elif command == "export":
                result = runner._single_state()
            elif command == "fingerprint":
                result = cloudpickle.loads(payload)(runner._single_state(branch=True))
            elif command == "evaluate":
                weights, scenario, steps, greedy, actions_only = payload
                outside = state()
                try:
                    if evaluator is None:
                        evaluator = cfg.env_runner_cls(config=cfg)
                    evaluator.module.set_state(weights)
                    result = evaluator.evaluate((scenario,), steps=steps, deterministic=greedy,
                                                actions_only=actions_only)[0]
                finally:
                    restore(outside)
            else:
                raise ValueError(f"Unknown resident command: {command}")
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:  # noqa: BLE001 -- propagate worker failures to the waiting controller.
        connection.send((False, traceback.format_exc()))
    finally:
        if evaluator is not None:
            evaluator.stop()
        if runner is not None:
            runner.stop()
        connection.close()


class ResidentExecution:
    def __init__(self, runner, saved):
        import cloudpickle
        import torch

        self.runner, self.owner = runner, uuid4().hex
        self.connections, self.processes, self.cursors = [], [], []
        self.token = 0
        workers = runner.config.execution_workers
        cfg = runner.config.copy(copy_frozen=False)
        cfg.execution_workers = 0
        config_blob = cloudpickle.dumps(cfg)
        seeds = {int(cfg.seed)} | {
            int(cfg.seed) * 1_000_003 + int(s)
            for s in (*cfg.train_seeds, *cfg.validation_seeds, *cfg.test_seeds)
        }
        cuda_seeds = {
            s: [torch.Generator(device=f"cuda:{i}").manual_seed(s).get_state()
                for i in range(torch.cuda.device_count())] or None for s in seeds
        }
        context = mp.get_context("spawn")
        environment = {key: os.environ.get(key) for key in
                       ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        try:
            os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
                              MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            for lane in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                    args=(child, config_blob, lane, cuda_seeds, torch.are_deterministic_algorithms_enabled()),
                    daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    self.close()
                    raise
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
        finally:
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        try:
            for i in range(len(self.connections)):
                self.receive(i)
            self.load(saved)
        except BaseException:
            self.close()
            raise

    def receive(self, index):
        connection, process = self.connections[index], self.processes[index]
        ready = wait([connection, process.sentinel])
        if connection not in ready:
            raise RuntimeError(f"Sampling worker {index} exited with code {process.exitcode}")
        try:
            success, result = connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"Sampling worker {index} disconnected") from exc
        if not success:
            raise RuntimeError(f"Sampling worker {index} failed:\n{result}")
        return result

    def broadcast(self, command, payloads):
        for connection, payload in zip(self.connections, payloads):
            connection.send((command, payload))
        return [self.receive(i) for i in range(len(payloads))]

    def load(self, saved):
        streams = saved["streams"]
        bank = list(streams["bank"])
        bank[streams["active_lane"]] = {k: v for k, v in saved.items() if k != "streams"}
        self.cursors = self.broadcast("load", [(s, streams["lane_steps"]) for s in bank])

    def collect(self, total_steps, deterministic):
        import cloudpickle
        import numpy as np
        from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

        from .runner import random_state, restore_random_state

        runner = self.runner
        count = len(runner.config.train_seed_groups)
        if total_steps < count or total_steps % count:
            raise ValueError("Phase samples must divide evenly across streams")
        steps, outside = total_steps // count, random_state()
        observer = getattr(runner, "sample_observer", None)
        observer = cloudpickle.dumps(observer) if observer is not None else None
        weights = runner.module.get_state()
        try:
            results = self.broadcast("collect", [
                (weights, steps, runner.trajectory_periods + lane * steps, runner.lane_steps,
                 outside, deterministic, observer) for lane in range(count)])
            totals = {}
            for _, meta, _, _ in results:
                for name, value in meta["last_metrics"].items():
                    totals[name] = totals.get(name, 0.0) + value
            meta = results[-1][1]
            meta.update(active_lane=count - 1, last_metrics=totals,
                        lane_steps=[n + steps for n in runner.lane_steps])
            runner._apply_sampling_metadata(meta)
            runner._discard_local_history()
            self.cursors = [item[2] for item in results]
            runner.last_sample_trace = [row for item in results for row in item[3]]
            return MultiAgentBatch({
                key: SampleBatch({name: np.concatenate([result[0][key][name] for result in results])
                                  for name in results[0][0][key]}) for key in runner.agent_ids
            }, total_steps)
        finally:
            restore_random_state(outside)

    def mode_returns(self, steps, options, seed):
        weights = self.runner.module.get_state()
        return self.broadcast("mode_returns", [
            (weights, steps, options, seed + lane * 1_000_000_007)
            for lane in range(len(self.runner.config.train_seed_groups))
        ])

    def snapshot(self):
        from .runner import random_state

        self.token += 1
        self.broadcast("save", [self.token] * len(self.runner.config.train_seed_groups))
        return BranchState(self.owner, self.token, self.runner._sampling_metadata(),
                           copy.deepcopy(self.cursors), random_state())

    def restore(self, state):
        from .runner import restore_random_state

        if state.owner != self.owner:
            raise ValueError("Branch state belongs to a closed or different worker set")
        self.broadcast("restore", [state.token] * len(self.runner.config.train_seed_groups))
        self.runner._apply_sampling_metadata(state.metadata)
        self.runner._discard_local_history()
        self.cursors = copy.deepcopy(state.cursors)
        restore_random_state(state.rng)

    def export(self):
        from .runner import random_state

        runner = self.runner
        count = len(runner.config.train_seed_groups)
        bank = self.broadcast("export", [None] * count)
        saved = dict(bank[runner.active_lane])
        saved.update(rng=random_state(), last_metrics=copy.deepcopy(runner.last_metrics),
                     sampled_periods=runner.trajectory_periods)
        bank[runner.active_lane] = copy.deepcopy(saved)
        saved["streams"] = {"seed_groups": runner.config.train_seed_groups,
                            "active_lane": runner.active_lane, "lane_steps": runner.lane_steps.copy(), "bank": bank}
        return saved

    def evaluate(self, policies, seeds, steps, deterministic, actions_only):
        jobs = [(weights, seed, steps, deterministic, actions_only) for weights in policies for seed in seeds]
        rows, count = [], len(self.connections)
        for start in range(0, len(jobs), count):
            rows.extend(self.broadcast("evaluate", jobs[start:start + count]))
        return [rows[i:i + len(seeds)] for i in range(0, len(rows), len(seeds))]

    def fingerprint(self, function):
        import cloudpickle

        return self.broadcast("fingerprint", [cloudpickle.dumps(function)] * len(self.runner.config.train_seed_groups))

    def release(self):
        self.broadcast("release", [None] * len(self.runner.config.train_seed_groups))

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, OSError):
                    pass
        for connection, process in zip(self.connections, self.processes):
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            connection.close()
