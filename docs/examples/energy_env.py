"""单智能体/MARL 共用的简化电池环境；不是 Ye 2021 论文环境复现。

控制对象只有电池，动作是 [-1, 1] 内的连续充放电指令。
正数表示充电，负数表示放电；一个 step 默认是半小时。
不依赖任何具体算法。EV、空调、P2P 市场等完整机制不包含在本示例中。
"""

from collections import deque

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class BatteryCommunity:
    """物理计算层：所有住户先共同执行动作，再统一计算容量惩罚与奖励。"""

    def __init__(self, config=None):
        config = config or {}
        self.horizon = int(config.get("horizon", 48))
        self.households = int(config.get("households", 2))
        self.history_len = int(config.get("history", 0))
        self.reward_mode = config.get("reward_mode", "cooperative")
        if self.horizon < 1 or self.households < 1 or self.history_len < 0:
            raise ValueError("horizon/households must be positive; history must be non-negative")
        if self.reward_mode not in ("cooperative", "individual"):
            raise ValueError("reward_mode must be cooperative or individual")
        self.agent_ids = [f"house_{i}" for i in range(self.households)]
        self.dt = 0.5  # 小时；功率 kW 乘以 dt 得到电量 kWh。
        self.capacity = 4.0  # 每个住户电池容量，kWh。
        self.max_power = 2.0  # 最大充放电功率，kW。
        self.efficiency = 0.95
        self.grid_limit = 1.5 * self.households  # 社区净进口容量，kW。
        self.histories = {aid: deque(maxlen=self.history_len) for aid in self.agent_ids}

    def observation_space(self):
        current = gym.spaces.Box(-1.0, 1.0, (8,), np.float32)
        if not self.history_len:
            return current
        return gym.spaces.Dict(
            {
                "current": current,
                "history": gym.spaces.Box(-1.0, 1.0, (self.history_len, 3), np.float32),
                "history_mask": gym.spaces.Box(0.0, 1.0, (self.history_len,), np.float32),
            }
        )

    def reset(self, rng):
        self.t = 0
        self.energy = np.full(self.households, self.capacity * 0.5, dtype=np.float64)
        self.initial_energy = self.energy.copy()
        # horizon 次动作需要 horizon+1 个状态，最后一个用于返回有效的最终观测。
        self.hours = np.arange(self.horizon + 1, dtype=np.float64) * self.dt % 24
        self.buy_price = 0.22 + 0.4 * ((self.hours >= 17) & (self.hours < 21))
        self.sell_price = np.full(self.horizon + 1, 0.10)
        self.load = np.empty((self.households, self.horizon + 1))
        self.pv = np.empty_like(self.load)
        for i in range(self.households):
            base = 1.2 + 0.1 * (i % 3)
            evening = 0.6 * np.exp(-(((self.hours - 19) / 3) ** 2))
            noise = rng.uniform(-0.05, 0.05, self.horizon + 1)
            self.load[i] = base + 0.3 * np.sin(self.hours / 24 * 2 * np.pi + i) + evening + noise
            self.pv[i] = (1.2 + 0.1 * (i % 3)) * np.maximum(np.sin((self.hours - 6) / 12 * np.pi), 0)
            self.histories[self.agent_ids[i]].clear()

    def current(self, i):
        """局部观测：SOC、负荷、光伏、买/卖电价、时间编码、剩余任务时间。"""
        angle = self.hours[self.t] / 24 * 2 * np.pi
        return np.array(
            [
                self.energy[i] / self.capacity,
                self.load[i, self.t] / 3,
                self.pv[i, self.t] / 3,
                self.buy_price[self.t],
                self.sell_price[self.t],
                np.sin(angle),
                np.cos(angle),
                (self.horizon - self.t) / self.horizon,
            ],
            dtype=np.float32,
        )

    def observe(self):
        result = {}
        for i, aid in enumerate(self.agent_ids):
            current = self.current(i)
            if not self.history_len:
                result[aid] = current
                continue
            # 历史只使用本住户过去的负荷、光伏和公开买电价，当前值另放 current。
            history = np.zeros((self.history_len, 3), dtype=np.float32)
            mask = np.zeros(self.history_len, dtype=np.float32)
            rows = list(self.histories[aid])
            if rows:
                history[: len(rows)] = np.stack(rows)
                mask[: len(rows)] = 1
            result[aid] = {"current": current, "history": history, "history_mask": mask}
        return result

    def advance(self, actions):
        if self.t >= self.horizon:
            raise RuntimeError("Episode ended: call reset before another step")
        if set(actions) != set(self.agent_ids):
            raise ValueError("Provide one action for every fixed household")
        # 保存 t 时刻的历史特征，再执行动作；返回 t+1 的观测时历史截止于 t。
        for i, aid in enumerate(self.agent_ids):
            if self.history_len:
                self.histories[aid].append(self.current(i)[[1, 2, 3]].copy())
        requested = np.array([float(np.asarray(actions[aid]).reshape(1)[0]) for aid in self.agent_ids])
        if not np.isfinite(requested).all():
            raise ValueError("Actions must be finite")
        requested = np.clip(requested, -1, 1) * self.max_power
        # 动作给出功率指令；实际执行量还受到 SOC 和充放电效率约束。
        max_charge = (self.capacity - self.energy) / (self.efficiency * self.dt)
        max_discharge = self.energy * self.efficiency / self.dt
        power = np.clip(
            requested, -np.minimum(self.max_power, max_discharge), np.minimum(self.max_power, max_charge)
        )
        self.energy += np.where(
            power >= 0, power * self.dt * self.efficiency, power * self.dt / self.efficiency
        )
        self.energy = np.clip(self.energy, 0, self.capacity)
        net = self.load[:, self.t] - self.pv[:, self.t] + power
        # 买电产生费用，卖电产生收入；净进口为负时按卖电价结算。
        energy_cost = (
            np.maximum(net, 0) * self.buy_price[self.t] - np.maximum(-net, 0) * self.sell_price[self.t]
        ) * self.dt
        degradation = 0.02 * np.abs(power) * self.dt
        excess = max(float(net.sum()) - self.grid_limit, 0)
        peak_penalty = 0.25 * excess**2 * self.dt
        self.t += 1
        terminated = self.t == self.horizon
        # 有限时域任务末尾要求补回初始库存，避免免费耗尽电池取得低电费。
        terminal_penalty = (
            2.0 * np.maximum(self.initial_energy - self.energy, 0)
            if terminated
            else np.zeros(self.households)
        )
        costs = energy_cost + degradation + peak_penalty / self.households + terminal_penalty
        rewards = {
            aid: -float(costs.sum() if self.reward_mode == "cooperative" else costs[i])
            for i, aid in enumerate(self.agent_ids)
        }
        infos = {
            aid: {
                "requested_power_kw": float(requested[i]),
                "executed_power_kw": float(power[i]),
                "soc": float(self.energy[i] / self.capacity),
                "net_grid_kw": float(net[i]),
                "energy_cost": float(energy_cost[i]),
                "degradation_cost": float(degradation[i]),
                "peak_penalty_share": peak_penalty / self.households,
                "terminal_penalty": float(terminal_penalty[i]),
                "individual_cost": float(costs[i]),
            }
            for i, aid in enumerate(self.agent_ids)
        }
        return self.observe(), rewards, terminated, infos


class SingleEnergyEnv(gym.Env):
    """单智能体适配：一个住户，返回普通 obs/action/reward 接口。"""

    def __init__(self, config=None):
        super().__init__()
        self.sim = BatteryCommunity({**(config or {}), "households": 1})
        self.observation_space = self.sim.observation_space()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(self.np_random)
        return self.sim.observe()["house_0"], {}

    def step(self, action):
        obs, rewards, terminated, infos = self.sim.advance({"house_0": action})
        # horizon 是任务本身的终点，且 remaining_time 已包含在观测中。
        return obs["house_0"], rewards["house_0"], terminated, False, infos["house_0"]


class MultiEnergyEnv(MultiAgentEnv):
    """多智能体适配：每个住户一个 agent，返回按 agent_id 索引的字典。"""

    def __init__(self, config=None):
        super().__init__()
        self.sim = BatteryCommunity(config)
        self.possible_agents = list(self.sim.agent_ids)
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.possible_agents)
        self.observation_spaces = {aid: self.sim.observation_space() for aid in self.possible_agents}
        self.action_spaces = {
            aid: gym.spaces.Box(-1.0, 1.0, (1,), np.float32) for aid in self.possible_agents
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agents = list(self.possible_agents)
        self.sim.reset(self.np_random)
        return self.sim.observe(), {aid: {} for aid in self.agents}

    def step(self, actions):
        obs, rewards, terminated, infos = self.sim.advance(actions)
        terms = {aid: terminated for aid in self.agents}
        truncs = {aid: False for aid in self.agents}
        return obs, rewards, {**terms, "__all__": terminated}, {**truncs, "__all__": False}, infos
