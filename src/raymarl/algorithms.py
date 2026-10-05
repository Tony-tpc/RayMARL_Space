"""RLlib Algorithm extensions for synchronized local multi-agent training."""

import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.checkpoints import Checkpointable
from ray.train._internal.storage import StorageContext

from .learner import JointTorchLearner
from .networks import JointModule
from .replay import JointReplay
from .spaces import ActionLayout


class JointConfig(AlgorithmConfig):
    variant = "maac"

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or JointAlgorithm)
        self.hidden_dim = 64
        self.actor_lr, self.critic_lr = 3e-4, 1e-3
        self.temperature, self.tau, self.gamma = 0.1, 0.001, 0.99
        self.batch_size, self.replay_capacity, self.warmup_steps = 128, 100_000, 512
        self.update_interval, self.baseline_samples = 4, 8
        self.counterfactual = True
        self.critic_mode = "attention"
        self.actor_grad_clip, self.critic_grad_clip = 0.5, 50.0
        self.max_episode_steps = 100_000
        self.exploration_std = 0.1
        self.storage_path = str(Path.cwd() / "ray_results")
        self.seed = 0
        self.num_env_runners = 0
        self.num_learners = 0
        self.num_gpus_per_learner = 0
        self.framework_str = "torch"

    def training(self, **kwargs):
        custom = {
            "hidden_dim",
            "temperature",
            "batch_size",
            "replay_capacity",
            "warmup_steps",
            "update_interval",
            "baseline_samples",
            "counterfactual",
            "critic_mode",
            "actor_grad_clip",
            "critic_grad_clip",
            "max_episode_steps",
            "exploration_std",
            "actor_lr",
            "critic_lr",
            "tau",
            "storage_path",
        }
        for name in list(kwargs):
            if name in custom:
                setattr(self, name, kwargs.pop(name))
        return super().training(**kwargs)

    def validate(self):
        if not self.enable_rl_module_and_learner or not self.enable_env_runner_and_connector_v2:
            raise ValueError("Only the RLlib new API stack is supported")
        if self.num_gpus_per_learner not in (0, 1):
            raise ValueError("v0.1 supports 0 or 1 local learner GPU")
        if self.num_env_runners != 0 or self.num_learners != 0:
            raise ValueError("Joint algorithms v0.1 require local EnvRunner/Learner (both counts 0)")
        if self.framework_str != "torch":
            raise ValueError("Only PyTorch is supported")
        for key in ("actor_lr", "critic_lr", "tau", "actor_grad_clip", "critic_grad_clip"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.tau > 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid tau/gamma")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        for key in (
            "hidden_dim",
            "batch_size",
            "replay_capacity",
            "warmup_steps",
            "update_interval",
            "baseline_samples",
            "max_episode_steps",
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.critic_mode not in ("attention", "uniform", "joint", "local"):
            raise ValueError("Unknown critic_mode")

    def build_algo(self, env=None, use_copy=True):
        cfg = self.copy(copy_frozen=False) if use_copy else self
        if env is not None:
            cfg.environment(env)
        cfg.validate()
        storage = StorageContext(
            storage_path=Path(cfg.storage_path).resolve().as_posix(),
            experiment_dir_name="raymarl",
            trial_dir_name=cfg.variant,
        )
        return cfg.algo_class(config=cfg, storage=storage)


class MAACConfig(JointConfig):
    pass


class MASACConfig(JointConfig):
    variant = "masac"

    def __init__(self, algo_class=None):
        super().__init__(algo_class)
        self.critic_mode = "joint"


class MADDPGConfig(JointConfig):
    variant = "maddpg"

    def __init__(self, algo_class=None):
        super().__init__(algo_class)
        self.critic_mode = "joint"
        self.temperature = 0.0


class ISACConfig(JointConfig):
    """Local soft critic sharing the hybrid kernels; not a copy of stock SAC."""

    variant = "isac"

    def __init__(self, algo_class=None):
        super().__init__(algo_class)
        self.critic_mode = "local"


class JointAlgorithm(Algorithm):
    @classmethod
    def get_default_config(cls):
        return MAACConfig(algo_class=cls)

    def setup(self, config):
        cfg = self.config
        self._set_up_metrics()
        classes = cfg.callbacks_class if isinstance(cfg.callbacks_class, list) else [cfg.callbacks_class]
        self.callbacks = [cls() for cls in classes if cls is not None]
        torch.manual_seed(cfg.seed or 0)
        np.random.seed(cfg.seed or 0)
        random.seed(cfg.seed or 0)
        self.joint_env = self.env_creator(EnvContext(cfg.env_config, worker_index=0))
        obs, _ = self.joint_env.reset(seed=cfg.seed)
        self.agent_order = tuple(sorted(obs, key=str))
        self.observation_spaces = [self.joint_env.get_observation_space(a) for a in self.agent_order]
        self.action_spaces = [self.joint_env.get_action_space(a) for a in self.agent_order]
        if cfg.variant == "maddpg" and any(not ActionLayout(a).is_continuous for a in self.action_spaces):
            self.joint_env.close()
            raise ValueError("Standard MADDPG requires continuous Box actions")
        module = JointModule(
            self.observation_spaces,
            self.action_spaces,
            cfg.hidden_dim,
            cfg.critic_mode,
            twin=cfg.variant == "masac",
        )
        self.joint_learner = JointTorchLearner(config=cfg, module=module)
        self.joint_learner.build()
        self._sampler_actors = [copy.deepcopy(a).cpu() for a in self.joint_learner.module.actors]
        self.replay = JointReplay(cfg.replay_capacity, cfg.seed)
        self.sampled_steps, self.episode_number = 0, 0
        self._initial_observation = obs
        self._at_boundary = True
        self._last_episode_return = None

    def get_module(self, module_id=None):
        if module_id is None:
            return self.joint_learner.module
        if module_id in self.joint_learner.module._rl_modules:
            return self.joint_learner.module[module_id]
        return self.joint_learner.module[f"agent_{self.agent_order.index(module_id)}"]

    def actions(self, observation, *, deterministic=False):
        result = {}
        with torch.no_grad():
            for aid, actor in zip(self.agent_order, self._sampler_actors):
                tensor = actor.encoder.layout.tensor([observation[aid]], "cpu")
                features = actor.distribution(tensor).draw(
                    deterministic=deterministic or self.config.variant == "maddpg"
                )[0]
                if not torch.isfinite(features).all():
                    raise FloatingPointError("Non-finite policy action")
                if self.config.variant == "maddpg" and not deterministic:
                    features = (features + torch.randn_like(features) * self.config.exploration_std).clamp(
                        -1, 1
                    )
                action = actor.layout.decode_features(features)

                def numpy_tree(x):
                    if isinstance(x, dict):
                        return {k: numpy_tree(v) for k, v in x.items()}
                    if isinstance(x, tuple):
                        return tuple(numpy_tree(v) for v in x)
                    a = x[0].cpu().numpy()
                    return a.item() if a.ndim == 0 else a

                result[aid] = numpy_tree(action)
        return result

    def _sync_sampler(self):
        for target, online in zip(self._sampler_actors, self.joint_learner.module.actors):
            target.load_state_dict({k: v.detach().cpu() for k, v in online.state_dict().items()})

    def _batch(self, rows):
        device = self.joint_learner._device
        actors = self.joint_learner.module.actors
        observations = [
            actor.encoder.layout.tensor([r[0][a] for r in rows], device)
            for a, actor in zip(self.agent_order, actors)
        ]
        next_observations = [
            actor.encoder.layout.tensor([r[3][a] for r in rows], device)
            for a, actor in zip(self.agent_order, actors)
        ]
        actions = [
            torch.stack([actor.layout.encode(r[1][a], device) for r in rows])
            for a, actor in zip(self.agent_order, actors)
        ]
        rewards = torch.tensor([[r[2][a] for a in self.agent_order] for r in rows], device=device)
        terminated = torch.tensor(
            [[r[4][a] for a in self.agent_order] for r in rows], device=device, dtype=torch.float32
        )
        return observations, actions, rewards, next_observations, terminated

    def training_step(self):
        cfg = self.config
        obs = self._initial_observation
        if obs is None:
            obs, _ = self.joint_env.reset(seed=(cfg.seed or 0) + self.episode_number)
        self._initial_observation = None
        self._at_boundary = False
        returns = {a: 0.0 for a in self.agent_order}
        losses = []
        for t in range(cfg.max_episode_steps):
            action = self.actions(obs)
            next_obs, reward, term, trunc, _info = self.joint_env.step(action)
            if set(reward) != set(self.agent_order):
                raise ValueError("Changing agent sets are unsupported in v0.1")
            terms = {a: bool(term.get(a, term.get("__all__", False))) for a in self.agent_order}
            truncs = {a: bool(trunc.get(a, trunc.get("__all__", False))) for a in self.agent_order}
            for a in self.agent_order:
                if a not in next_obs:
                    if not terms[a]:
                        raise ValueError("Missing bootstrap observation for a non-terminal agent")
                    next_obs[a] = obs[a]
                returns[a] += float(reward[a])
            self.replay.add(obs, action, reward, next_obs, terms, truncs, self.episode_number, t)
            self.sampled_steps += 1
            if self.sampled_steps >= cfg.warmup_steps and self.sampled_steps % cfg.update_interval == 0:
                losses.append(
                    self.joint_learner.update_joint(*self._batch(self.replay.sample(cfg.batch_size)))
                )
                self._sync_sampler()
            obs = next_obs
            if term.get("__all__", False) or trunc.get("__all__", False):
                break
            if any(terms.values()) or any(truncs.values()):
                raise ValueError("Partial agent termination is unsupported in v0.1")
        else:
            raise RuntimeError("Environment did not finish within max_episode_steps")
        self.episode_number += 1
        self._at_boundary = True
        self._last_episode_return = returns
        mean_losses = {k: float(np.mean([x[k] for x in losses])) for k in losses[0]} if losses else {}
        return {
            "num_env_steps_sampled_lifetime": self.sampled_steps,
            "env_runners": {
                "episode_return_mean": float(np.mean(list(returns.values()))),
                "episode_returns_per_agent": returns,
                "episode_len_mean": t + 1,
            },
            "learners": mean_losses,
            "num_updates_lifetime": self.joint_learner.steps,
            "learner_device": str(self.joint_learner._device),
        }

    def step(self):
        return self.training_step()

    def get_checkpointable_components(self):
        return []

    def get_state(self, *args, **kwargs):
        if not self._at_boundary:
            raise RuntimeError("Save checkpoints only at completed episode boundaries")
        return copy.deepcopy(
            {
                "agent_order": self.agent_order,
                "spaces": (self.observation_spaces, self.action_spaces),
                "learner": self.joint_learner.get_state(),
                "replay": self.replay.get_state(),
                "sampled_steps": self.sampled_steps,
                "episode_number": self.episode_number,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if self.joint_learner._device.type == "cuda"
                else None,
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            }
        )

    def set_state(self, state):
        if tuple(state["agent_order"]) != self.agent_order or state["spaces"] != (
            self.observation_spaces,
            self.action_spaces,
        ):
            raise ValueError("Checkpoint agent/space layout mismatch")
        if state["replay"]["capacity"] != self.replay.capacity:
            raise ValueError("Checkpoint replay capacity mismatch")
        self.joint_learner.set_state(state["learner"])
        self._sync_sampler()
        self.replay.set_state(state["replay"])
        self.sampled_steps, self.episode_number = state["sampled_steps"], state["episode_number"]
        self._initial_observation = None
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])

    def restore_from_path(self, path, *args, **kwargs):
        return Checkpointable.restore_from_path(self, path, *args, **kwargs)

    def export_actors(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observation_spaces": self.observation_spaces,
                "action_spaces": self.action_spaces,
                "hidden_dim": self.config.hidden_dim,
                "agent_order": self.agent_order,
                "actors": [
                    {k: v.detach().cpu().clone() for k, v in a.state_dict().items()}
                    for a in self.joint_learner.module.actors
                ],
            },
            path,
        )

    def cleanup(self):
        if hasattr(self, "joint_env"):
            self.joint_env.close()
