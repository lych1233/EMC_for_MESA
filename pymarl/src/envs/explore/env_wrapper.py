import numpy as np

from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
from copy import deepcopy

from .MPE_env import MPEEnv


class ExploreWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs # ["env_args"]
        
        if isinstance(args, dict):
            args = convert(args)

        self.env = MPEEnv(args)
        self.n_agents = args.num_agents
        self.episode_limit = args.episode_length
        self.reset()
        self.unit_dim = self.get_obs_size()

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self.env.reset()
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Returns reward, terminated, info """
        self._obs, reward, done, info = self.env.step([np.eye(5)[a] for a in actions])
        return reward[0][0], done[0], {"success": 1. if reward[0][0] > 0.99 else 0.}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return deepcopy(self._obs)

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        return deepcopy(np.concatenate(self._obs))

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        avail_actions = [np.ones(5) for _ in range(self.n_agents)]
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(5)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return 5

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, x):
        self.env.seed(x)