"""Native two-stage episode contracts only. No Algorithm, Learner or trainer."""
from dataclasses import astuple

import numpy as np
import pytest

pytest.importorskip("ray")
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

from snmarket import StageError
from snmarket.action import map_consumer_controls, map_uc_controls
from snmarket.rllib_env import SnEnergyMarketEnv

pytestmark = pytest.mark.integration


def valid_actions(env):
    engine = env.engine
    if engine.phase.value == "UC_TURN":
        t = engine.state.period_id
        action = map_uc_controls([0, 0.8, 0, 0.2, 0.2, 0, 0.3], engine.state,
                                  engine.profiles.der_kw[t], engine.config)
        return {"uc": np.array(astuple(action), dtype=np.float64)}
    return {i: np.array(astuple(map_consumer_controls([0.2, 0.5, 0.5, 0.5], bounds)))
            for i, bounds in engine.local_bounds().items()}


def test_native_episode_phase_clock_reward_and_terminal_contract(make_engine):
    engine = make_engine([[4, 5]] * 5, flex=2, der=3, deadline=2)
    env = SnEnergyMarketEnv(dict(config=engine.config, profiles=engine.profiles))
    obs, infos = env.reset(seed=1)
    assert set(obs) == {"uc"}
    episode = MultiAgentEpisode(observation_space=env.observation_space, action_space=env.action_space)
    episode.add_env_reset(observations=obs, infos=infos)
    for call in range(10):
        action = valid_actions(env)
        for i, value in action.items():
            assert env.action_spaces[i].contains(value)
        obs, rewards, terms, truncs, infos = env.step(action)
        episode.add_env_step(observations=obs, actions=action, rewards=rewards, infos=infos,
                             terminateds=terms, truncateds=truncs)
        for i, value in obs.items():
            assert env.observation_spaces[i].contains(value), (i, value)
            assert len(value) == (22 if i == "uc" else 26)
        assert set(infos) == set(obs)
        assert not any(truncs.values())
        assert env.engine.state.period_id == (call + 1) // 2
        if call % 2 == 0:
            assert set(obs) == set(engine.consumer_ids)
            assert rewards == {}
        else:
            assert set(rewards) == set(env.possible_agents)
            if call < 9:
                assert set(obs) == {"uc"}
                assert not terms["__all__"]
    assert terms["__all__"]
    assert set(obs) == set(env.possible_agents)
    assert all(infos[i]["terminal_observation"] for i in env.possible_agents)
    assert env.agents == []
    episode.validate()
    assert episode.is_terminated
    assert len(episode) == 10
    assert all(len(item) == 5 for item in episode.agent_episodes.values())
    with pytest.raises(StageError):
        env.step({})
    env.close()


def test_uc_info_keeps_only_aggregate_feedback_and_negative_cash_in_space(make_engine):
    engine = make_engine([[4], [4]], grid_cost=0.8)
    env = SnEnergyMarketEnv(dict(config=engine.config, profiles=engine.profiles))
    env.reset()
    env.step({"uc": np.array([4, 0, 0, 0, 0, 0], dtype=float)})
    obs, _, _, _, infos = env.step({"consumer_0": np.array([0, 0, 0, 0.3])})
    assert obs["uc"][-1] < 0
    assert env.observation_spaces["uc"].contains(obs["uc"])
    assert not {"state", "services", "consumers", "tasks", "monthly", "central_state"}.intersection(infos["uc"])


def test_adapter_snapshot_restores_acting_agents(make_engine):
    engine = make_engine([[4], [4]])
    env = SnEnergyMarketEnv(dict(config=engine.config, profiles=engine.profiles))
    env.reset()
    beginning = env.get_state()
    env.step(valid_actions(env))
    pending = env.get_state()
    assert env.agents == ["consumer_0"]
    env.set_state(beginning)
    assert env.agents == ["uc"]
    env.set_state(pending)
    assert env.agents == ["consumer_0"]


def test_rectangular_action_space_does_not_replace_joint_validation(make_engine):
    engine = make_engine([4], der=5)
    env = SnEnergyMarketEnv(dict(config=engine.config, profiles=engine.profiles))
    env.reset()
    simultaneous = np.array([4, 0, 1, 0, 1, 0], dtype=float)
    assert env.action_spaces["uc"].contains(simultaneous)
    before = env.get_state()
    with pytest.raises(ValueError):
        env.step({"uc": simultaneous})
    assert env.get_state() == before
