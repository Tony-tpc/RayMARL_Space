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


def generalized_advantages(rewards, values, next_values, terminated, gamma, lambda_gae, segment_end=None):
    """One discount per physical period; segment ends bootstrap but stop recursion."""
    rewards, values, next_values = [np.asarray(x, dtype=np.float64) for x in (rewards, values, next_values)]
    terminated = np.asarray(terminated, dtype=bool)
    boundaries = np.zeros(len(rewards), dtype=bool) if segment_end is None else np.asarray(segment_end, dtype=bool)
    advantages = np.empty_like(rewards)
    carry = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuation = float(not terminated[index])
        delta = rewards[index] + gamma * continuation * next_values[index] - values[index]
        carry = delta + gamma * lambda_gae * continuation * (not boundaries[index]) * carry
        advantages[index] = carry
    return advantages, advantages + values


def bootstrapped_returns(rewards, next_values, terminated, gamma, critic_lambda, segment_end=None):
    """Fixed batch lambda-returns, with no recursion across real episode ends."""
    targets = np.empty(len(rewards), dtype=np.float64)
    carry = float(next_values[-1])
    boundaries = np.zeros(len(rewards), dtype=bool) if segment_end is None else np.asarray(segment_end, dtype=bool)
    for index in range(len(rewards) - 1, -1, -1):
        weight = critic_lambda * (not boundaries[index])
        tail = 0.0 if terminated[index] else (
            (1 - weight) * next_values[index] + weight * carry
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
        self._execution = None
        self._remote_dirty = False
        self.sample_observer = None
        self.last_sample_trace = []
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
        return tuple(self.module.keys())

    @property
    def env(self):
        # Direct environment inspection is supported, but normal resident sampling
        # and cursor diagnostics never materialize the history in the controller.
        if self._remote_dirty:
            self._remote_dirty = False
            metadata = self._sampling_metadata()
            saved = self._execution.export()
            self._bank = saved["streams"]["bank"]
            self._load_single(saved, preserve_learner_rng=True)
            self._apply_sampling_metadata(metadata)
        return self._env

    @env.setter
    def env(self, value):
        self._env = value

    def _sampling_metadata(self):
        return copy.deepcopy({
            "observation": self._observation, "terminated": self._terminated,
            "data_exhausted": self._data_exhausted, "scenario_seed": self._scenario_seed,
            "train_seed_index": self._train_seed_index, "last_metrics": self.last_metrics,
            "sampled_periods": self.trajectory_periods, "active_lane": self.active_lane,
            "lane_steps": self.lane_steps,
        })

    def _apply_sampling_metadata(self, metadata):
        for key in ("observation", "terminated", "data_exhausted", "scenario_seed", "train_seed_index"):
            setattr(self, "_" + key, copy.deepcopy(metadata[key]))
        self.last_metrics = copy.deepcopy(metadata["last_metrics"])
        self.num_env_steps_sampled_lifetime = self.trajectory_periods = metadata["sampled_periods"]
        self.active_lane = metadata["active_lane"]
        self.lane_steps = metadata["lane_steps"].copy()

    def lane_summary(self):
        return {"lane": self.active_lane, "seed": self._scenario_seed,
                "steps": self.lane_steps[self.active_lane]}

    def _start_execution(self):
        if self._execution is None and getattr(self.config, "execution_workers", 0):
            from .resident import ResidentExecution

            self._execution = ResidentExecution(self, self.get_state(components="sampling")["sampling"])
            self._discard_local_history()
        return self._execution

    def _discard_local_history(self):
        self._bank = [None] * len(self.config.train_seed_groups)
        self._env = None
        self._remote_dirty = True

    def branch_state(self):
        if self._execution is None:
            return self.get_state()
        return {COMPONENT_RL_MODULE: copy.deepcopy(self.module.get_state()),
                "weights_seq_no": self._weights_seq_no, "sampling": self._execution.snapshot()}

    def release_branches(self):
        if self._execution is not None:
            self._execution.release()

    def sampling_fingerprint(self, function):
        if self._execution is None:
            return function(self.get_state(components="sampling"))
        return function(self._sampling_metadata()), self._execution.fingerprint(function)

    def _serial_sampling(self):
        if self._execution is not None:
            saved = self._execution.export()
            self._execution.close()
            self._execution = None
            self._remote_dirty = False
            self.set_state({"sampling": saved})

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
        target = 0 if self.config.phase_sampling == "all_streams" else outer_round % len(self._bank)
        if self._execution is not None and self.config.phase_sampling == "all_streams":
            return self.active_lane
        self._switch_lane(target)
        return self.active_lane

    def _switch_lane(self, target):
        self._serial_sampling()
        self._bank[self.active_lane] = self._single_state()
        if target != self.active_lane:
            total = self.trajectory_periods
            self.active_lane = target
            self._load_single(dict(self._bank[target], sampled_periods=total), preserve_learner_rng=True)

    def append_profiles(self, profiles):
        self._serial_sampling()
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

    @torch.no_grad()
    def _values(self, key, inputs):
        """Each module owns its value network; batch only its saved time samples."""
        device = next(self.module[key].parameters()).device
        inputs = torch.as_tensor(np.asarray(inputs), device=device, dtype=torch.float32)
        return self.module[key].compute_values({"obs": inputs, "critic_obs": inputs}).cpu().numpy().astype(np.float64)

    def _announce(self, deterministic):
        bounds = self.env.uc_bounds()
        sample = self._draw("uc", self._observation, bounds, deterministic=deterministic)
        decision = self.env.publish(np.r_[sample["storage_mode"], sample["raw_latents"]])
        sample["actions"] = decision.uc_action
        sample["mapping_context"] = np.asarray(list(decision.mapping_context.values()), dtype=np.float64)
        sample.update(
            obs=decision.uc_observation.copy(),
            critic_obs=decision.uc_value_state.copy(),
            bounds=bounds,
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
        rng = random_state()
        try:
            with self._temporary_announcement():
                _, decision = self._announce(deterministic)
                return {
                    "uc": decision.uc_value_state.copy(),
                    **{key: decision.consumer_value_state.copy() for key in self.env.consumer_ids},
                }
        finally:
            restore_random_state(rng)

    @contextmanager
    def _temporary_announcement(self):
        temporary = getattr(self.env, "temporary_announcement", None)
        if temporary is not None:
            with temporary():
                yield
        else:
            snapshot = self.env.get_state()
            try:
                yield
            finally:
                self.env.set_state(snapshot)

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

    def _period(self, *, deterministic=False, actions_only=False, with_values=True):
        with self.action_stream():
            result = self._sample_period(deterministic=deterministic, actions_only=actions_only, with_values=with_values)
        self.lane_steps[self.active_lane] += 1
        return result

    def _sample_period(self, *, deterministic=False, actions_only=False, with_values=True):
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
                obs=observation.copy(),
                critic_obs=decision.consumer_value_state.copy(),
                bounds=bounds,
            )
            samples[key] = sample
            actions[key] = sample["actions"]
        result = self.env.step(actions)
        self._observation = result.next_uc_observation
        self._terminated, self._data_exhausted = result.terminated, result.data_exhausted
        self.num_env_steps_sampled_lifetime += 1
        self.trajectory_periods += 1
        for key, sample in samples.items():
            if with_values:
                sample["vf_preds"] = self._value(key, sample["obs"], sample["critic_obs"])
            sample.update(
                rewards=result.rewards[key],
                terminateds=result.terminated,
                truncateds=False,
                period_id=decision.period_id,
                scenario_seed=self._scenario_seed,
            )
        return samples, result

    def collect(self, steps, *, deterministic=False):
        self._serial_sampling()
        rows = {key: [] for key in self.agent_ids}
        totals = {}
        for _ in range(steps):
            if self._terminated:
                self._start_next_episode()
            samples, result = self._period(deterministic=deterministic, with_values=False)
            for key, sample in samples.items():
                if rows[key] and not rows[key][-1]["terminateds"]:
                    rows[key][-1]["next_value_obs"] = sample["critic_obs"].copy()
                sample["stream_id"] = self.active_lane
                sample["segment_end"] = bool(sample["terminateds"])
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
            samples[-1]["truncateds"] = not samples[-1]["terminateds"]
            samples[-1]["segment_end"] = True
            data = {name: np.asarray([sample[name] for sample in samples]) for name in samples[0]}
            inputs = data["critic_obs"]
            if key in next_inputs:
                inputs = np.concatenate((inputs, next_inputs[key][None]), axis=0)
            predictions = self._values(key, inputs)
            data["vf_preds"] = predictions[:steps]
            data["next_vf_preds"] = np.r_[predictions[1:steps], predictions[-1] if key in next_inputs else 0.0]
            data["next_vf_preds"][data["terminateds"]] = 0.0
            data["advantages"], _ = generalized_advantages(
                data["rewards"],
                data["vf_preds"],
                data["next_vf_preds"],
                data["terminateds"],
                self.config.gamma,
                self.config.lambda_gae,
                data["segment_end"],
            )
            # Separate fixed targets: lambda=1 uses realized rewards until the
            # batch cut (old-value bootstrap) or true termination (zero tail).
            data["value_targets"] = bootstrapped_returns(
                data["rewards"], data["next_vf_preds"], data["terminateds"],
                self.config.gamma, self.config.critic_lambda,
                data["segment_end"],
            )
            data["value_bootstrap"] = bootstrapped_returns(
                np.zeros_like(data["rewards"]), data["next_vf_preds"], data["terminateds"],
                self.config.gamma, self.config.critic_lambda,
                data["segment_end"],
            )
            batches[key] = SampleBatch(data)
        self.last_metrics = totals
        return MultiAgentBatch(batches, steps)

    def collect_phase(self, total_steps, *, deterministic=False):
        """Collect a fixed-policy phase; total_steps is never multiplied by stream count."""
        if self.config.phase_sampling == "single_stream":
            return self.collect(total_steps, deterministic=deterministic)
        execution = self._start_execution()
        if execution is not None:
            return execution.collect(total_steps, deterministic)
        count = len(self._bank)
        if total_steps < count or total_steps % count:
            raise ValueError("Phase samples must divide evenly across nonempty streams")
        parts, totals = [], {}
        for lane in range(count):
            self._switch_lane(lane)
            parts.append(self.collect(total_steps // count, deterministic=deterministic))
            for name, value in self.last_metrics.items():
                totals[name] = totals.get(name, 0.0) + value
        self.last_metrics = totals
        return MultiAgentBatch({
            key: SampleBatch({name: np.concatenate([part[key][name] for part in parts])
                              for name in parts[0][key]}) for key in self.agent_ids
        }, total_steps)

    def sample(self, *, num_timesteps=None, **kwargs):
        return self.collect(num_timesteps or self.config.rollout_length)

    def evaluate(self, seeds, *, steps=None, deterministic=False, actions_only=False):
        """Realized finite-episode scores; optional windows are diagnostics only."""
        execution = self._start_execution()
        if execution is not None:
            return execution.evaluate([self.module.get_state()], tuple(seeds), steps, deterministic, actions_only)[0]
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

    def evaluate_candidates(self, policies, seeds, *, steps=None):
        """All candidate/month jobs share the same bounded CPU worker set."""
        execution = self._start_execution()
        if execution is not None:
            return execution.evaluate(policies, tuple(seeds), steps, False, True)
        weights = copy.deepcopy(self.module.get_state())
        try:
            results = []
            for policy in policies:
                self.module.set_state(policy)
                results.append(self.evaluate(seeds, steps=steps, actions_only=True))
            return results
        finally:
            self.module.set_state(weights)

    def get_metrics(self):
        self.metrics.log_value(NUM_ENV_STEPS_SAMPLED_LIFETIME, self.num_env_steps_sampled_lifetime, window=1)
        return self.metrics.reduce()

    def _single_state(self, *, branch=False):
        environment_state = getattr(self.env, "branch_state", self.env.get_state) if branch else self.env.get_state
        return {
            "environment": environment_state(),
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

    def _load_single(self, saved, *, preserve_learner_rng=False, branch=False):
        outside = random_state()
        self.restart(saved["scenario_seed"])
        restore = getattr(self.env, "restore_branch", self.env.set_state) if branch else self.env.set_state
        restore(saved["environment"])
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
            if self._execution is not None:
                state["sampling"] = self._execution.export()
                return state
            state["sampling"] = self._single_state()
            bank = copy.deepcopy(self._bank)
            bank[self.active_lane] = copy.deepcopy(state["sampling"])
            state["sampling"]["streams"] = {
                "seed_groups": self.config.train_seed_groups,
                "active_lane": self.active_lane,
                "lane_steps": self.lane_steps.copy(),
                "bank": bank,
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
            from .resident import BranchState

            if isinstance(saved, BranchState):
                if self._execution is None:
                    raise ValueError("Resident branch requires its original live workers")
                self._execution.restore(saved)
                return
            streams = saved["streams"]
            if tuple(map(tuple, streams["seed_groups"])) != self.config.train_seed_groups:
                raise ValueError("Sampling stream groups differ from the configured training groups")
            self.active_lane = streams["active_lane"]
            self.lane_steps = streams["lane_steps"].copy()
            self._bank = copy.deepcopy(streams["bank"])
            if self._execution is not None:
                self._execution.load(saved)
            self._remote_dirty = False
            self._load_single(saved)

    def get_ctor_args_and_kwargs(self):
        return (), {"config": self.config, "worker_index": self.worker_index}

    def stop(self):
        if self._execution is not None:
            self._execution.close()
            self._execution = None
        self._remote_dirty = False
        self.env = None
