import numpy as np

from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
from copy import deepcopy

from .one_step import ClimbGame_Single_10
from .K_one import K_one
from .multi_step import ClimbGame_Multi_X


class NaiveWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs # ["env_args"]
        
        if isinstance(args, dict):
            args = convert(args)

        if args.scenario_name == "one_step_10":
            self.env = ClimbGame_Single_10(args.seed)
        if args.scenario_name == "K_one":
            self.env = K_one(args.seed)
        elif args.scenario_name == "multi_step_3":
            self.env = ClimbGame_Multi_X(args.seed, _n_act=3)
        elif args.scenario_name == "multi_step_10":
            self.env = ClimbGame_Multi_X(args.seed, _n_act=10)
        
        self.n_agents = args.num_agents
        self.episode_limit = args.episode_length
        self.reset()

        self._n_act = self.env._n_act
        self.unit_dim = self.get_obs_size()

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self.env.reset()
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Returns reward, terminated, info """
        self._obs, reward, done, info = self.env.step([[a] for a in actions])
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
        avail_actions = [np.ones(self._n_act) for _ in range(self.n_agents)]
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self._n_act)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self._n_act

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, x):
        self.env.seed(x)