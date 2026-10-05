"""Centralized-information reference, never mixed with decentralized claims."""
import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from city_proxy import CityLearnProxy


class CentralReference(MultiAgentEnv):
    def __init__(self,config):
        super().__init__();self.base=CityLearnProxy(config)
        self.possible_agents=['central'];self.agents=self.possible_agents;self._agent_ids={'central'}
        self.action_spaces={'central':gym.spaces.Dict(self.base.action_spaces)}
        n=len(self.base.agents)
        d=sum(self.base.observation_spaces[a]['current'].shape[0] for a in self.base.agents)
        self.observation_spaces={'central':gym.spaces.Dict({
            'current':gym.spaces.Box(-np.inf,np.inf,(d,),np.float32),
            'history':gym.spaces.Box(-np.inf,np.inf,(48,3*n),np.float32),
            'history_mask':gym.spaces.Box(0,1,(48,),np.float32)})}
        self.native_returns={}

    def wrap(self,o):
        return {'central':{'current':np.concatenate([o[a]['current'] for a in self.base.agents]),
            'history':np.concatenate([o[a]['history'] for a in self.base.agents],axis=-1),
            'history_mask':o[self.base.agents[0]]['history_mask']}}

    def reset(self,seed=None,options=None):
        o,info=self.base.reset(seed=seed);self.native_returns={a:0. for a in self.base.agents}
        return self.wrap(o),{'central':{}}

    def step(self,action):
        o,r,t,tr,i=self.base.step(action['central'])
        for a in r:self.native_returns[a]+=float(r[a])
        return self.wrap(o),{'central':sum(r.values())}, {'__all__':t['__all__']},{'__all__':tr['__all__']},{'central':{}}

    def evaluate(self):return self.base.evaluate()
    def close(self):self.base.close()
