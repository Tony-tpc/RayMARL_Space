"""Small native RLlib modules and synchronized episode fixtures."""

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

from .fixtures import InterfaceGame


class HistoryGame(InterfaceGame):
    def __init__(self, config=None):
        super().__init__(config)
        self.observation_spaces = {
            aid: gym.spaces.Dict(
                {
                    "current": space,
                    "history": gym.spaces.Box(-1, 1, (3, 2), np.float32),
                    "history_mask": gym.spaces.Box(0, 1, (3,), np.float32),
                }
            )
            for aid, space in self.observation_spaces.items()
        }

    def _obs(self):
        return {
            aid: {
                "current": np.full(space["current"].shape, self.t / self.horizon, np.float32),
                "history": np.zeros((3, 2), np.float32),
                "history_mask": np.array([1, 1, 0], np.float32),
            }
            for aid, space in self.observation_spaces.items()
        }


def make_learner(config_cls, *, hybrid=True, history=False, gpu=0):
    config = config_cls().environment(
        HistoryGame if history else InterfaceGame, env_config={"hybrid": hybrid}
    )
    config.rl_module(model_config={"hidden_dim": 8})
    config.learners(num_learners=0, num_gpus_per_learner=gpu)
    config.training(batch_size=4, warmup_steps=4)
    if hasattr(config, "baseline_samples"):
        config.training(baseline_samples=2)
    config.prepare_policies()
    env = (HistoryGame if history else InterfaceGame)({"hybrid": hybrid})
    spec = config.get_multi_rl_module_spec(env=env, inference_only=False)
    learner = config.get_default_learner_class()(config=config, module_spec=spec)
    learner.build()
    episodes = make_episodes(env, config.policy_mapping_fn)
    return learner, episodes


def make_episodes(env, mapping, *, terminated=True, truncated=False, count=4):
    episodes = []
    for i in range(count):
        obs, _ = env.reset(seed=i)
        actions = {aid: env.get_action_space(aid).sample() for aid in env.possible_agents}
        next_obs, rewards, _, _, _ = env.step(actions)
        episodes.append(
            MultiAgentEpisode(
                id_=f"fixture-{i}",
                observations=[obs, next_obs],
                actions=[actions],
                rewards=[rewards],
                terminateds={**{aid: terminated for aid in env.possible_agents}, "__all__": terminated},
                truncateds={**{aid: truncated for aid in env.possible_agents}, "__all__": truncated},
                observation_space=env.observation_spaces,
                action_space=env.action_spaces,
                agent_to_module_mapping_fn=mapping,
                len_lookback_buffer=0,
            )
        )
    return episodes
