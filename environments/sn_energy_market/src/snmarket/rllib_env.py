"""Two-stage RLlib adapter. It carries physical actions and never constructs a policy."""
from dataclasses import asdict

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .config import EnvironmentConfig, individual
from .contracts import Phase
from .engine import EnergyMarketEngine, StageError
from .observation import consumer_fields, uc_fields
from .profiles import load_profiles


class SnEnergyMarketEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        values = dict(env_config or {})
        if set(values) - {"config", "config_dir", "profiles", "profile_path"}:
            raise ValueError("Unknown environment options")
        if "config" in values and "config_dir" in values:
            raise ValueError("Choose config or config_dir")
        config = (EnvironmentConfig.from_directory(values["config_dir"]) if "config_dir" in values
                  else values.get("config", EnvironmentConfig()))
        if not isinstance(config, EnvironmentConfig):
            raise ValueError("config must be an EnvironmentConfig")
        if "profiles" in values and "profile_path" in values:
            raise ValueError("Choose profiles or profile_path")
        profiles = load_profiles(values["profile_path"], config) if "profile_path" in values else values.get("profiles")
        self.engine = EnergyMarketEngine(config, profiles)
        super().__init__()
        self.possible_agents = ["uc", *self.engine.consumer_ids]
        self.agents = ["uc"]
        self._agent_ids = set(self.possible_agents)
        self._build_spaces()

    def _build_spaces(self):
        s, m, p = self.engine.config.system, self.engine.config.market, self.engine.profiles
        continuous = s.episode_mode == "continuous"
        der_max = np.inf if continuous else float(p.der_kw.max())
        upper = np.array([s.grid_import_max_kw, der_max, s.ess_charge_max_kw,
                          s.ess_charge_max_kw, s.ess_discharge_max_kw, m.dr_procurement_max_kw])
        self.action_spaces = {"uc": gym.spaces.Box(np.zeros(6), upper, dtype=np.float64)}
        uc_low = np.zeros(m.shift_deadline_steps + 20, dtype=np.float32)
        uc_low[1] = -1 if continuous else 0
        uc_low[-1] = -np.inf
        self.observation_spaces = {"uc": gym.spaces.Box(uc_low, np.full_like(uc_low, np.inf))}
        maximum_tariff = max(m.uc_flat_rates + m.uc_high_rates + m.uc_peak_rates)
        for i, identity in enumerate(self.engine.consumer_ids):
            flex = np.inf if continuous else float(p.flex_kw[:, i].max())
            lo = np.array([0, -flex, 0, m.der_reference_rate_per_kwh], dtype=np.float64)
            hi = np.array([flex, flex, 1, min(individual(m.der_bid_cap, i), maximum_tariff)], dtype=np.float64)
            self.action_spaces[identity] = gym.spaces.Box(lo, hi, dtype=np.float64)
            low = np.zeros(m.shift_deadline_steps + 24, dtype=np.float32)
            low[1] = -1 if continuous else 0
            self.observation_spaces[identity] = gym.spaces.Box(low, np.full_like(low, np.inf))
        self.action_space = gym.spaces.Dict(self.action_spaces)
        self.observation_space = gym.spaces.Dict(self.observation_spaces)

    def _infos(self, observations, record=None):
        result = {}
        terminal = self.engine.phase == Phase.TERMINATED
        for identity in observations:
            info = {"phase": self.engine.phase.value, "physical_period": self.engine.state.period_id,
                    "terminal_observation": terminal, "data_exhausted": self.engine.data_exhausted,
                    "bootstrap_observation_valid": not terminal and not self.engine.data_exhausted,
                    "schema_version": 9, "mechanism_version": self.engine.config.market.mechanism_version}
            if record:
                info.update(transition_id=record.transition_id, reward_period=record.period_id)
                if identity == "uc":
                    info.update(cash_profit=record.ledger.cash_profit, uc_reward=record.ledger.uc_reward,
                                supply_shortfall_kw=record.ledger.supply_shortfall_kw,
                                power_residual_kw=record.physical.power_residual_kw)
                else:
                    info["service"] = asdict(record.services[identity])
                    info["ledger"] = asdict(record.ledger.consumers[identity])
            result[identity] = info
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        observations = {"uc": self.engine.reset(seed=seed, options=options)}
        self.agents = ["uc"]
        return observations, self._infos(observations)

    def step(self, action_dict):
        if self.engine.phase == Phase.TERMINATED:
            raise StageError("Episode has terminated; call reset")
        if self.engine.phase == Phase.UC_TURN:
            if set(action_dict) != {"uc"}:
                raise ValueError("UC phase accepts only the uc action")
            observations = self.engine.apply_uc_action(action_dict["uc"])
            self.agents = list(self.engine.consumer_ids)
            return observations, {}, {"__all__": False}, {"__all__": False}, self._infos(observations)
        if set(action_dict) != set(self.engine.consumer_ids):
            raise ValueError("Consumer phase requires exactly all consumer actions")
        record = self.engine.apply_consumer_actions(action_dict)
        observations = {"uc": self.engine.uc_observation()}
        if record.terminated:
            observations.update(self.engine.consumer_observations(terminal=True))
            self.agents = []
            terms = dict.fromkeys(self.possible_agents, True)
            terms["__all__"] = True
        else:
            self.agents = ["uc"]
            terms = {"__all__": False}
        return observations, record.rewards, terms, dict.fromkeys(terms, False), self._infos(observations, record)

    def get_state(self):
        return self.engine.get_state()

    def set_state(self, snapshot):
        self.engine.set_state(snapshot)
        self.agents = (["uc"] if self.engine.phase == Phase.UC_TURN else
                       list(self.engine.consumer_ids) if self.engine.phase == Phase.CONSUMER_TURN else [])

    def append_profiles(self, profiles):
        self.engine.append_profiles(profiles)

    def central_state(self):
        """Trusted diagnostics only; UC actor/value receives its aggregate observation."""
        return self.engine.central_state()

    def close(self):
        pass

    @property
    def observation_fields(self):
        h = self.engine.config.market.shift_deadline_steps
        return {"uc": uc_fields(h), **{i: consumer_fields(h) for i in self.engine.consumer_ids}}

