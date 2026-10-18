"""Uniform joint replay keyed by episode and environment timestep."""

import copy

import numpy as np
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.utils.checkpoints import Checkpointable


class JointReplay(Checkpointable):
    def __init__(self, capacity=100_000, seed=0):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity, self.seed = capacity, seed
        self.rows, self.position = [], 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, reward, next_observation, terminated, truncated, episode_id, env_t):
        keys = set(observation)
        if any(set(v) != keys for v in (action, reward, next_observation, terminated, truncated)):
            raise ValueError("Joint transition must include every fixed agent")
        row = copy.deepcopy(
            (observation, action, reward, next_observation, terminated, truncated, episode_id, env_t)
        )
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.position] = row
        self.position = (self.position + 1) % self.capacity

    def add_episodes(self, episodes, agent_order):
        expected = set(agent_order)
        for episode in episodes:
            if set(episode.agent_episodes) != expected or any(
                len(sa) != len(episode) or sa.t_started != episode.env_t_started
                for sa in episode.agent_episodes.values()
            ):
                raise ValueError("Dynamic agents and asynchronous episode transitions are unsupported")
            for t in range(len(episode)):
                # Synchronous single-agent episode indices are local to the chunk.
                # This avoids the hanging-reward/global-index ambiguity in cut episodes.
                obs = {aid: episode.agent_episodes[aid].get_observations(t) for aid in agent_order}
                actions = {aid: episode.agent_episodes[aid].get_actions(t) for aid in agent_order}
                rewards = {aid: episode.agent_episodes[aid].get_rewards(t) for aid in agent_order}
                next_obs = {aid: episode.agent_episodes[aid].get_observations(t + 1) for aid in agent_order}
                last = t == len(episode) - 1
                term = episode.get_terminateds()
                trunc = episode.get_truncateds()
                terms = {aid: bool(last and term.get(aid, False)) for aid in agent_order}
                truncs = {aid: bool(last and trunc.get(aid, False)) for aid in agent_order}
                if set(obs) != expected or set(actions) != expected or set(rewards) != expected:
                    raise ValueError(
                        "Dynamic agents, asynchronous steps and partial termination are unsupported"
                    )
                if any(terms.values()) and not all(terms.values()):
                    raise ValueError("Partial agent termination is unsupported")
                if any(truncs.values()) and not all(truncs.values()):
                    raise ValueError("Partial agent truncation is unsupported")
                for aid in agent_order:
                    if aid not in next_obs:
                        if not terms[aid]:
                            raise ValueError("Missing bootstrap observation for a non-terminal agent")
                        next_obs[aid] = obs[aid]
                self.add(
                    obs, actions, rewards, next_obs, terms, truncs, episode.id_, episode.env_t_started + t
                )

    def sample(self, size):
        if not self.rows:
            raise ValueError("Cannot sample empty replay")
        return [self.rows[i] for i in self.rng.integers(0, len(self.rows), size=size)]

    def sample_episodes(self, size, mapping, observation_spaces, action_spaces):
        episodes = []
        for index, row in enumerate(self.sample(size)):
            obs, actions, rewards, next_obs, terms, truncs, eid, t = row
            episodes.append(
                MultiAgentEpisode(
                    id_=f"replay-{eid}-{t}-{index}",
                    observations=[copy.deepcopy(obs), copy.deepcopy(next_obs)],
                    actions=[copy.deepcopy(actions)],
                    rewards=[copy.deepcopy(rewards)],
                    terminateds={**terms, "__all__": all(terms.values())},
                    truncateds={**truncs, "__all__": all(truncs.values())},
                    observation_space=observation_spaces,
                    action_space=action_spaces,
                    agent_to_module_mapping_fn=mapping,
                    len_lookback_buffer=0,
                )
            )
        return episodes

    def get_state(self, *args, **kwargs):
        return copy.deepcopy(
            {
                "capacity": self.capacity,
                "rows": self.rows,
                "position": self.position,
                "rng": self.rng.bit_generator.state,
            }
        )

    def set_state(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity mismatch")
        self.rows, self.position = copy.deepcopy(state["rows"]), state["position"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def get_ctor_args_and_kwargs(self):
        return (), {"capacity": self.capacity, "seed": self.seed}

    def get_checkpointable_components(self):
        return []

    def __len__(self):
        return len(self.rows)
