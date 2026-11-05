"""Current training configuration and native lifecycle helpers."""
import copy
import hashlib
import os
import random
import secrets
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.sn_energy_market.profile_dataset import (
    DatasetEnvFactory,
    ProfileDataset,
)
from experiments.sn_energy_market.settings import (
    inventory_valued_config,
    model_settings,
)
from raymarl.algorithms.stackelberg.algorithm import Stackelberg
from raymarl.algorithms.stackelberg.config import StackelbergConfig
from raymarl.algorithms.stackelberg.runner import StackelbergEnvRunner

ROOT = Path(__file__).resolve().parents[2]
TRIAL_SEED = 11
SCREEN_SEEDS = (10042, 10048)
MONTH_SEEDS = tuple(range(10042, 10054))
EVALUATION_SEEDS = (110001, 110002, 110003)


@contextmanager
def ray_session(directory, *, device="cuda"):
    import ray

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(directory)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    token = directory / "auth_token"
    if not token.exists():
        token.write_text(secrets.token_hex(32), encoding="utf-8")
    os.environ["RAY_AUTH_MODE"] = "token"
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(token)
    torch.set_num_threads(1)
    num_gpus = {"cpu": 0, "cuda": 1}[device]
    if num_gpus and not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ray.init(num_cpus=2, num_gpus=num_gpus, include_dashboard=False,
             _node_ip_address="127.0.0.1", _temp_dir=str(directory / "ray"), log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


def digest_tree(value):
    """Content hash, independent of pickle storage identities and tensor devices."""
    h = hashlib.sha256()

    def visit(x):
        if hasattr(x, "fingerprint"):
            h.update(str(x.fingerprint()).encode())
        elif torch.is_tensor(x):
            visit(x.detach().cpu().numpy())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode())
            h.update(x.tobytes())
        elif is_dataclass(x):
            h.update(type(x).__qualname__.encode())
            for f in fields(x):
                h.update(f.name.encode())
                visit(getattr(x, f.name))
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                visit(k)
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(str((type(x).__name__, len(x))).encode())
            for item in x:
                visit(item)
        elif isinstance(x, (set, frozenset)):
            visit(sorted(x, key=str))
        else:
            h.update((type(x).__name__ + ":" + repr(x)).encode())

    visit(value)
    return h.hexdigest()


class TrainingRunner(StackelbergEnvRunner):
    """Energy cursor diagnostics; sampling and checkpoint logic live in the native runner."""

    def lane_cursors(self):
        if self._execution is not None:
            return copy.deepcopy(self._execution.cursors)
        rows = []
        for i, saved in enumerate(self._bank):
            state = (
                self.env.engine.state
                if i == self.active_lane
                else saved["environment"].engine.state
            )
            seed = (
                self._scenario_seed if i == self.active_lane else saved["scenario_seed"]
            )
            rows.append(
                {
                    "lane": i,
                    "seed": seed,
                    "period": state.period_id,
                    "timestamp": state.timestamp.isoformat(),
                    "steps": self.lane_steps[i],
                    "soc_kwh": state.ess_energy_kwh,
                }
            )
        return rows

    def lane_summary(self):
        state = self.env.engine.state
        return {"lane": self.active_lane, "seed": self._scenario_seed,
                "period": state.period_id, "timestamp": state.timestamp.isoformat(),
                "steps": self.lane_steps[self.active_lane], "soc_kwh": state.ess_energy_kwh}


class TrainingAlgorithm(Stackelberg):
    def get_state(self, *args, **kwargs):
        state = super().get_state(*args, **kwargs)
        audit = getattr(self, "experiment_audit", None)
        state["experiment_audit"] = (
            audit.state() if audit else getattr(self, "restored_audit", {})
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        self.restored_audit = copy.deepcopy(state.get("experiment_audit", {}))
        if getattr(self, "experiment_audit", None):
            self.experiment_audit.load(self.restored_audit)


class TrainingConfig(StackelbergConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or TrainingAlgorithm)
        self.env_runner_cls = TrainingRunner


def configuration(output, *, device="cuda"):
    dataset = ProfileDataset(ROOT / "datasets/ausgrid/processed/manifest.json",
                             splits=("train", "validation"))
    factory = DatasetEnvFactory(dataset, inventory_valued_config(dataset.environment_config()), seed=42)
    return (TrainingConfig()
        .scenarios(factory, train_seeds=factory.seed_splits["train"],
                   train_seed_groups=factory.training_seed_groups,
                   validation_seeds=SCREEN_SEEDS, test_seeds=())
        .debugging(seed=TRIAL_SEED)
        .rl_module(model_config={**model_settings(128), "online_popart": True,
                                 "critic_hidden_dims": {"uc": 192},
                                 "critic_input_scales": {"uc": {0: 1488.0, 1: 1488.0}}})
        .multi_agent(algorithm_config_overrides_per_module={"uc": {"actor_lr": 1e-4}})
        .learners(num_learners=0, num_gpus_per_learner={"cpu": 0, "cuda": 1}[device])
        .training(execution_workers=4, phase_sampling="all_streams",
                  gamma=0.997, lambda_gae=0.97, critic_lambda=1.0, uc_frozen_gae=True,
                  uc_critic_replay_batches=12,
                  rollout_length=384, inner_updates=4, response_updates=4,
                  uc_candidates=1, uc_updates_per_candidate=4, deviation_updates=0,
                  critic_epochs=2, num_epochs=1, minibatch_size=32,
                  always_keep_candidate=True, acceptance_z=1.0,
                  evaluation_steps=None, stopping_patience=0,
                  storage_path=str(output), checkpoint_interval=0, keep_checkpoints=2))


def initialize_fresh(algorithm):
    """Generate seed-11 weights explicitly, after RLlib/runner construction."""
    torch.manual_seed(TRIAL_SEED)
    np.random.seed(TRIAL_SEED)
    random.seed(TRIAL_SEED)
    module = algorithm.config.get_multi_rl_module_spec().build()
    # Match the offline summary initialization: new features start with zero weights.
    with torch.no_grad():
        module["uc"].value.encoder[0].weight[:, -8:].zero_()
    weights = module.get_state()
    for result in algorithm.learner_group.foreach_learner(
        lambda learner: learner.module.set_state(weights)
    ):
        result.get()
    for item in algorithm.learner_group.foreach_learner(
        lambda learner: [len(opt.state) for opt in learner._named_optimizers.values()]
    ):
        assert not any(item.get()), "Fresh initialization must not contain Adam moments"
    algorithm._sync()
    return digest_tree(algorithm.learner_group.get_weights())
