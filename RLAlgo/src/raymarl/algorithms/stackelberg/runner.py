"""Physical-period sampling for the leader announcement and follower response."""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME

from .contracts import TrainingEnvironment


def random_state():
    return {
        "torch": torch.get_rng_state().clone(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def generalized_advantages(rewards, values, next_values, terminated, gamma, lambda_gae):
    """One discount per physical period; segment ends bootstrap but stop recursion."""
    rewards, values, next_values = [np.asarray(x, dtype=np.float64) for x in (rewards, values, next_values)]
    terminated = np.asarray(terminated, dtype=bool)
    advantages = np.empty_like(rewards)
    carry = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuation = float(not terminated[index])
        delta = rewards[index] + gamma * continuation * next_values[index] - values[index]
        carry = delta + gamma * lambda_gae * continuation * carry
        advantages[index] = carry
    return advantages, advantages + values


def bootstrapped_returns(rewards, next_values, terminated, gamma, critic_lambda):
    """Fixed batch lambda-returns, with no recursion across real episode ends."""
    targets = np.empty(len(rewards), dtype=np.float64)
    carry = float(next_values[-1])
    for index in range(len(rewards) - 1, -1, -1):
        tail = 0.0 if terminated[index] else (
            (1 - critic_lambda) * next_values[index] + critic_lambda * carry
        )
        carry = rewards[index] + gamma * tail
        targets[index] = carry
    return targets


class StackelbergEnvRunner(EnvRunner, Checkpointable):
    """A local native EnvRunner whose rows each represent a complete physical period.

    Sampling always leaves the real environment before the next UC announcement.
    A temporary announcement supplies a segment-end Consumer bootstrap value; its
    environment and random state are then restored, so an update cannot leave a stale
    UC action pending for the next batch.
    """

    env: TrainingEnvironment

    def __init__(self, *, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self._train_seed_index = 0
        self._scenario_seed = int(self.config.train_seed_groups[0][0])
        self._weights_seq_no = 0
        # RLlib can broadcast its aggregate counter between train() calls.
        # This snapshot-owned cursor counts only the committed sampling branch.
        self.trajectory_periods = 0
        self.active_lane = 0
        self.lane_steps = [0] * len(self.config.train_seed_groups)
        self._bank = []
        self.restart(self._scenario_seed)
        self.module = self.config.get_multi_rl_module_spec(
            spaces=self.get_spaces(), inference_only=False
        ).build()
        self.last_metrics = {}
        first = self._single_state()
        for group in self.config.train_seed_groups:
            self.restart(group[0])
            self._bank.append(self._single_state())
        self._load_single(first)

    def assert_healthy(self):
        assert self.env is not None and self.module is not None

    def get_spaces(self):
        box = lambda size, dtype=np.float32: gym.spaces.Box(-np.inf, np.inf, (size,), dtype)
        return {
            "uc": (box(self.env.uc_observation_dim), box(6, np.float64)),
            **{
                key: (box(self.env.consumer_observation_dim), box(4, np.float64))
                for key in self.env.consumer_ids
            },
        }

    @property
    def agent_ids(self):
        return ("uc", *self.env.consumer_ids)

    def restart(self, seed):
        """Start an explicit scenario while retaining lifetime sampling counters."""
        self._scenario_seed = int(seed)
        if seed in self.config.train_seeds:
            self._train_seed_index = self.config.train_seeds.index(seed)
        self.env = self.config.scenario_factory(self._scenario_seed)
        self._observation = self.env.reset(self._scenario_seed)
        self._terminated = self._data_exhausted = False
        outside = random_state()
        try:
            seed = int(self.config.seed) * 1_000_003 + self._scenario_seed
            torch.manual_seed(seed)
            np.random.seed(seed % 2**32)
            random.seed(seed)
            self._action_rng = random_state()
        finally:
            restore_random_state(outside)

    def _start_next_episode(self):
        seeds = self.config.train_seed_groups[self.active_lane]
        self.restart(seeds[(seeds.index(self._scenario_seed) + 1) % len(seeds)])

    @contextmanager
    def action_stream(self):
        """Sampling owns its RNG; learner shuffles cannot change paused trajectories."""
        outside = random_state()
        restore_random_state(self._action_rng)
        try:
            yield
        finally:
            self._action_rng = random_state()
            restore_random_state(outside)

    def begin_round(self, outer_round):
        """Switch only between complete outer rounds; each stream resumes in place."""
        self._bank[self.active_lane] = self._single_state()
        target = outer_round % len(self._bank)
        if target != self.active_lane:
            total = self.trajectory_periods
            self.active_lane = target
            self._load_single(dict(self._bank[target], sampled_periods=total), preserve_learner_rng=True)
        return self.active_lane

    def append_profiles(self, profiles):
        self._observation = self.env.append_profiles(profiles)
        self._data_exhausted = False

    def _tensor(self, key, value, *, dtype=torch.float32):
        device = next(self.module[key].parameters()).device
        return torch.as_tensor(value, device=device, dtype=dtype).unsqueeze(0)

    def _draw(self, key, observation, bounds=None, deterministic=False):
        with torch.no_grad():
            output = self.module[key].draw(
                self._tensor(key, observation),
                None if bounds is None else self._tensor(key, bounds, dtype=torch.float64),
                deterministic=deterministic,
            )
        return {name: value.detach().cpu().numpy()[0].copy() for name, value in output.items()}

    def _value(self, key, observation, critic_observation):
        with torch.no_grad():
            value = self.module[key].compute_values(
                {
                    "obs": self._tensor(key, observation),
                    "critic_obs": self._tensor(key, critic_observation),
                }
            )
        return float(value.item())

    def _announce(self, deterministic):
        bounds = self.env.uc_bounds()
        sample = self._draw("uc", self._observation, bounds, deterministic=deterministic)
        decision = self.env.publish(np.r_[sample["storage_mode"], sample["raw_latents"]])
        sample["actions"] = decision.uc_action
        sample["mapping_context"] = np.asarray(list(decision.mapping_context.values()), dtype=np.float64)
        sample.update(
            obs=decision.uc_observation,
            critic_obs=decision.uc_observation,
            bounds=bounds,
            vf_preds=self._value("uc", decision.uc_observation, decision.uc_observation),
        )
        return sample, decision

    def bootstrap_inputs(self, *, deterministic=False):
        with self.action_stream():
            return self._bootstrap_inputs(deterministic=deterministic)

    def _bootstrap_inputs(self, *, deterministic=False):
        """Save next value inputs, including the original temporary UC announcement."""
        if self._terminated:
            return {}
        if self._data_exhausted:
            raise RuntimeError("Continuous data exhausted: append profiles before value bootstrapping")
        snapshot, rng = self.env.get_state(), random_state()
        try:
            _, decision = self._announce(deterministic)
            return {
                "uc": decision.uc_observation.copy(),
                **{
                    key: decision.consumer_value_state.copy()
                    for key in self.env.consumer_ids
                },
            }
        finally:
            self.env.set_state(snapshot)
            restore_random_state(rng)

    def bootstrap_values(self, *, deterministic=False):
        """Estimate both next decision values without consuming an action or RNG draw."""
        inputs = self.bootstrap_inputs(deterministic=deterministic)
        return {key: self._value(key, inputs[key], inputs[key]) if key in inputs else 0.0
                for key in self.agent_ids}

    def _action_period(self, deterministic):
        """Same action draws and RNG order; candidate scores need no value/logp."""
        with torch.no_grad():
            controls = self.module["uc"].draw_action(
                self._tensor("uc", self._observation),
                self._tensor("uc", self.env.uc_bounds(), dtype=torch.float64), deterministic,
            )[0].cpu().numpy()
            decision = self.env.publish(controls)
            actions = {
                key: self.module[key].draw_action(
                    self._tensor(key, decision.consumer_observations[key]),
                    self._tensor(key, decision.consumer_bounds[key], dtype=torch.float64), deterministic,
                )[0].cpu().numpy()
                for key in self.env.consumer_ids
            }
        return {}, self.env.step(actions)

    def _period(self, *, deterministic=False, actions_only=False):
        with self.action_stream():
            result = self._sample_period(deterministic=deterministic, actions_only=actions_only)
        self.lane_steps[self.active_lane] += 1
        return result

    def _sample_period(self, *, deterministic=False, actions_only=False):
        if self._data_exhausted:
            raise RuntimeError("Continuous data exhausted: append profiles before collecting another period")
        if actions_only:
            samples, result = self._action_period(deterministic)
            self._observation = result.next_uc_observation
            self._terminated, self._data_exhausted = result.terminated, result.data_exhausted
            self.num_env_steps_sampled_lifetime += 1
            self.trajectory_periods += 1
            return samples, result
        uc_sample, decision = self._announce(deterministic)
        samples = {"uc": uc_sample}
        actions = {}
        for key in self.env.consumer_ids:
            observation = decision.consumer_observations[key]
            bounds = decision.consumer_bounds[key]
            sample = self._draw(key, observation, bounds, deterministic)
            sample.update(
                obs=observation,
                critic_obs=decision.consumer_value_state,
                bounds=bounds,
                vf_preds=self._value(key, observation, decision.consumer_value_state),
            )
            samples[key] = sample
            actions[key] = sample["actions"]
        result = self.env.step(actions)
        self._observation = result.next_uc_observation
        self._terminated, self._data_exhausted = result.terminated, result.data_exhausted
        self.num_env_steps_sampled_lifetime += 1
        self.trajectory_periods += 1
        for key, sample in samples.items():
            sample.update(
                rewards=result.rewards[key],
                terminateds=result.terminated,
                truncateds=False,
                period_id=decision.period_id,
                scenario_seed=self._scenario_seed,
            )
        return samples, result

    def collect(self, steps, *, deterministic=False):
        rows = {key: [] for key in self.agent_ids}
        totals = {}
        for _ in range(steps):
            if self._terminated:
                self._start_next_episode()
            samples, result = self._period(deterministic=deterministic)
            for key, sample in samples.items():
                if rows[key] and not rows[key][-1]["terminateds"]:
                    rows[key][-1]["next_vf_preds"] = sample["vf_preds"]
                    rows[key][-1]["next_value_obs"] = sample["critic_obs"].copy()
                sample["next_vf_preds"] = 0.0
                # UC uses its local observation; consumers use the centralized
                # value input after the sampled UC announcement. Terminals stay zero.
                sample["next_value_obs"] = np.zeros_like(sample["critic_obs"])
                rows[key].append(sample)
            for name, value in result.metrics.items():
                totals[name] = totals.get(name, 0.0) + value
        next_inputs = self.bootstrap_inputs(deterministic=deterministic)
        batches = {}
        for key, samples in rows.items():
            if key in next_inputs:
                samples[-1]["next_value_obs"] = next_inputs[key]
                samples[-1]["next_vf_preds"] = self._value(key, next_inputs[key], next_inputs[key])
            samples[-1]["truncateds"] = not samples[-1]["terminateds"]
            data = {name: np.asarray([sample[name] for sample in samples]) for name in samples[0]}
            data["advantages"], _ = generalized_advantages(
                data["rewards"],
                data["vf_preds"],
                data["next_vf_preds"],
                data["terminateds"],
                self.config.gamma,
                self.config.lambda_gae,
            )
            # Separate fixed targets: lambda=1 uses realized rewards until the
            # batch cut (old-value bootstrap) or true termination (zero tail).
            data["value_targets"] = bootstrapped_returns(
                data["rewards"], data["next_vf_preds"], data["terminateds"],
                self.config.gamma, self.config.critic_lambda,
            )
            data["value_bootstrap"] = bootstrapped_returns(
                np.zeros_like(data["rewards"]), data["next_vf_preds"], data["terminateds"],
                self.config.gamma, self.config.critic_lambda,
            )
            batches[key] = SampleBatch(data)
        self.last_metrics = totals
        return MultiAgentBatch(batches, steps)

    def sample(self, *, num_timesteps=None, **kwargs):
        return self.collect(num_timesteps or self.config.rollout_length)

    def evaluate(self, seeds, *, steps=None, deterministic=False, actions_only=False):
        """Realized finite-episode scores; optional windows are diagnostics only."""
        state = self.get_state()
        rows = []
        try:
            for seed in seeds:
                self.restart(seed)
                returns = {key: 0.0 for key in self.agent_ids}
                totals, period = {}, 0
                while not self._terminated and (steps is None or period < steps):
                    _, result = self._period(deterministic=deterministic, actions_only=actions_only)
                    for key, reward in result.rewards.items():
                        returns[key] += self.config.gamma**period * reward
                    for key, value in result.metrics.items():
                        totals[key] = totals.get(key, 0.0) + value
                    period += 1
                bootstrap = self.bootstrap_values(deterministic=deterministic)
                estimated_returns = {
                    key: value + self.config.gamma**period * bootstrap[key] for key, value in returns.items()
                }
                native = totals.get("native_requested_kwh", 0.0)
                rows.append(
                    {
                        "seed": seed,
                        "returns": returns,
                        "bootstrap_estimated_returns": estimated_returns,
                        "bootstrap_values": bootstrap,
                        "terminated": self._terminated,
                        "score_complete": self._terminated,
                        "terminal_energy_value": totals.get("terminal_energy_value", 0.0),
                        "initial_energy_kwh": totals.get("initial_energy_kwh", 0.0),
                        "final_energy_kwh": totals.get("final_energy_kwh", 0.0),
                        "cash_profit": totals.get("uc_cash", 0.0),
                        "native_lost_kwh": totals.get("native_lost_kwh", 0.0),
                        "served_kwh": totals.get("served_kwh", 0.0),
                        "consumer_cost": totals.get("consumer_cost", 0.0),
                        "total_dr_payment": totals.get("dr_payment", 0.0),
                        "service_rate": 1.0 - totals.get("native_lost_kwh", 0.0) / native if native else 1.0,
                        "physical_periods": period,
                        "bootstrap_used": not self._terminated,
                        "metrics": totals,
                    }
                )
        finally:
            self.set_state(state)
        return rows

    def get_metrics(self):
        self.metrics.log_value(NUM_ENV_STEPS_SAMPLED_LIFETIME, self.num_env_steps_sampled_lifetime, window=1)
        return self.metrics.reduce()

    def _single_state(self):
        return {
            "environment": self.env.get_state(),
            "observation": copy.deepcopy(self._observation),
            "terminated": self._terminated,
            "data_exhausted": self._data_exhausted,
            "scenario_seed": self._scenario_seed,
            "train_seed_index": self._train_seed_index,
            "rng": random_state(),
            "action_rng": copy.deepcopy(self._action_rng),
            "last_metrics": copy.deepcopy(self.last_metrics),
            "sampled_periods": self.trajectory_periods,
        }

    def _load_single(self, saved, *, preserve_learner_rng=False):
        outside = random_state()
        self.restart(saved["scenario_seed"])
        self.env.set_state(saved["environment"])
        self._observation = copy.deepcopy(saved["observation"])
        self._terminated, self._data_exhausted = saved["terminated"], saved["data_exhausted"]
        self._train_seed_index = saved["train_seed_index"]
        self.last_metrics = copy.deepcopy(saved["last_metrics"])
        self.num_env_steps_sampled_lifetime = self.trajectory_periods = saved["sampled_periods"]
        self._action_rng = copy.deepcopy(saved["action_rng"])
        restore_random_state(outside if preserve_learner_rng else saved["rng"])

    def get_state(self, components=None, *, not_components=None, **kwargs):
        state = {}
        if self._check_component(COMPONENT_RL_MODULE, components, not_components):
            state[COMPONENT_RL_MODULE] = copy.deepcopy(self.module.get_state())
            state["weights_seq_no"] = self._weights_seq_no
        if self._check_component("sampling", components, not_components):
            state["sampling"] = self._single_state()
            state["sampling"]["streams"] = {
                "seed_groups": self.config.train_seed_groups,
                "active_lane": self.active_lane,
                "lane_steps": self.lane_steps.copy(),
                "bank": copy.deepcopy(self._bank),
            }
        return state

    def set_state(self, state):
        if COMPONENT_RL_MODULE in state:
            self.module.set_state(state[COMPONENT_RL_MODULE])
            self._weights_seq_no = state.get("weights_seq_no", 0)
        if NUM_ENV_STEPS_SAMPLED_LIFETIME in state:
            self.num_env_steps_sampled_lifetime = state[NUM_ENV_STEPS_SAMPLED_LIFETIME]
        if "sampling" in state:
            saved = state["sampling"]
            streams = saved["streams"]
            if tuple(map(tuple, streams["seed_groups"])) != self.config.train_seed_groups:
                raise ValueError("Sampling stream groups differ from the configured training groups")
            self.active_lane = streams["active_lane"]
            self.lane_steps = streams["lane_steps"].copy()
            self._bank = copy.deepcopy(streams["bank"])
            self._load_single(saved)

    def get_ctor_args_and_kwargs(self):
        return (), {"config": self.config, "worker_index": self.worker_index}

    def stop(self):
        self.env = None
