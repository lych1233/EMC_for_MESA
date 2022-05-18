from copy import copy, deepcopy

import gym
import numpy as np
from pettingzoo import ParallelEnv
from multiagent_mujoco.mujoco_multi import MujocoMulti

from envs.multiagentenv import MultiAgentEnv


class MAMuJoCoEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs # ["env_args"]
        
        if isinstance(args, dict):
            from utils.dict2namedtuple import convert
            args = convert(args)

        self.num_agents = args.num_agents
        self.num = args.num_angle_tasks
        self.angle_range = args.max_task_angles
        np.random.seed(args.seed + 1357)
        if args.scenario_name == "swimmer":
            assert self.num_agents == 2
            env_config = {
                'seed': self.seed,
                'reward_type': 0,
                'task_space': np.random.uniform(-self.angle_range / 180 * np.pi, self.angle_range / 180 * np.pi, size=self.num)
            }
            env_config["current_task"] = env_config['task_space'][0]
            self.env = SwimmerWrapper(env_config)
        elif args.scenario_name == "swimmer_velocity":
            assert self.num_agents == 2
            env_config = {
                'seed': self.seed,
                'reward_type': 1,
                'task_space': np.random.uniform(-self.angle_range / 180 * np.pi, self.angle_range / 180 * np.pi, size=self.num)
            }
            env_config["current_task"] = env_config['task_space'][0]
            self.env = SwimmerWrapper(env_config)
        else:
            raise NotImplementedError
        
        """self.observation_space = self._tl(self.env.observation_spaces)
        self.action_space = self._tl(self.env.action_spaces)
        total_dims = 0
        for obs_space in self.observation_space:
            total_dims += obs_space.shape[0]
        self.share_observation_space = [gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_dims, )) for _ in range(self.num_agents)]
        """

        self.n_agents = self.num_agents
        self.episode_limit = 1000
        self.reset()
        self.unit_dim = self.get_obs_size()
    
    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._reset()
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Returns reward, terminated, info """
        self._obs, reward, done, info = self._step([np.array([np.linspace(-1, 1, 11)[a]]) for a in actions])
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
        avail_actions = [np.ones(11) for _ in range(self.n_agents)]
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(11)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return 11

    def get_stats(self):
        return None
    
    def _tl(self, x):
        return [x[k] for k in self.env.agents]
    
    def _td(self, x):
        return {k: x[i] for i, k in enumerate(self.env.agents)}
    
    def _reset(self):
        return self._tl(self.env.reset())
    
    def _step(self, a):
        o, r, d, i = self.env.step(self._td(a))
        r_n = self._tl(r)
        r_n = [[x] for x in r_n]
        return self._tl(o), r_n, self._tl(d), [{} for _ in range(self.num_agents)]
    
    def render(self, mode):
        return self.env.render(mode)
    
    def close(self):
        pass
    
    def seed(self, seed=None):
        np.random.seed(seed)
        self.env.seed = seed
        self.env.config["seed"] = seed

class SwimmerWrapper(ParallelEnv):
    EPSILON = 2e-2
    RANGE = 0.5

    def __init__(self, env_config):
        env_args = {"scenario": "Swimmer-v2",
                    "agent_conf": "2x1",
                    "agent_obsk": 1,
                    "episode_limit": 1000}

        self.env = MujocoMulti(env_args=env_args)
        env_info = self.env.get_env_info()
        n_agents = env_info["n_agents"]

        self.agents = [f"player_{i}" for i in range(n_agents)]
        self.possible_agents = copy(self.agents)
        self.observation_spaces = {
            self.agents[i]: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env_info['obs_shape'],))
            for i, agent in enumerate(self.agents)}
        self.action_spaces = {
            self.agents[i]: env_info['action_spaces'][i] for i, agent in enumerate(self.agents)
        }

        self.all_tasks = env_config['task_space']
        self.seed = env_config['seed']
        self.current_task = env_config.get('current_task', None)
        self.config = copy(env_config)
        # from policies.shared import logging
        # self.logging = logging

    def reset(self):
        self.env.reset()
        obs = self.env.get_obs()
        return {agent: obs[i] for i, agent in enumerate(self.agents)}

    def step(self, action):
        # print(action)
        action = [action[agent] for agent in self.agents]
        reward, done, _ = self.env.step(action)
        obs = self.env.get_obs()

        # assert self.config['reward_type'] == 0
        if self.config['reward_type'] == 0:
            reward = 1.0
        elif self.config['reward_type'] == 1:
            reward = 1.0 if reward > self.RANGE else 0.0
        else:
            reward = reward

        # Recalculate reward based on current_task
        if self.current_task is None:
            # raise ValueError("No task selected")
            ret = [{agent: obs[i] for i, agent in enumerate(self.agents)},
                   {agent: 0 for i, agent in enumerate(self.agents)},
                   {agent: done for i, agent in enumerate(self.agents)},
                   {agent: None for i, agent in enumerate(self.agents)}]
            return ret

        state = self.env.get_state()
        angles = state[1:3]
        # check if two angles meet the task
        correct = [abs(angles[0] - self.current_task) < self.EPSILON, abs(angles[1] - self.current_task) < self.EPSILON]
        if sum(correct) == 2:
            reward *= 1
            self.logging.info('YAY!')
        elif sum(correct) == 0:
            reward *= 0.5
        else:
            reward *= 0

        ret = [{agent: obs[i] for i, agent in enumerate(self.agents)},
               {agent: reward for i, agent in enumerate(self.agents)},
               {agent: done for i, agent in enumerate(self.agents)},
               {agent: None for i, agent in enumerate(self.agents)}]
        return ret

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def max_num_agents(self):
        return len(self.possible_agents)

    def __str__(self):
        return "Multi_Swimmer_v2"

    @property
    def unwrapped(self):
        return self.env

    def render(self, mode="human"):
        return [self.env.env.render(mode=mode)]


"""class SwimmerWrapper(ParallelEnv):
    EPSILON = 2e-2
    RANGE = 1e-2

    # Generating goal in Swimmer
    # def generate_task(env, seed, num):
    #     if env == 'Swimmer':
    #         np.random.seed(seed)
    #         angle_range = shared.args['max_angle'] # Set as 30
    #         return np.random.uniform(-angle_range / 180 * math.pi, angle_range / 180 * math.pi, size=num)
    #     return None

    # Task space is set to contain 5 tasks for each run, [t_0, ... t_4]

    # Env can be created by PettingZoo: ParallelPettingZooEnv(SwimmerWrapper(env_config))
    
    def __init__(self, env_config):
        env_args = {"scenario": "Swimmer-v2",
                    "agent_conf": "2x1",
                    "agent_obsk": 1,
                    "episode_limit": 1000}

        self.env = MujocoMulti(env_args=env_args)
        env_info = self.env.get_env_info()
        n_agents = env_info["n_agents"]

        self.agents = [f"player_{i}" for i in range(n_agents)]
        self.possible_agents = copy(self.agents)
        self.observation_spaces = {
            self.agents[i]: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env_info['obs_shape'],))
            for i, agent in enumerate(self.agents)}
        self.action_spaces = {
            self.agents[i]: env_info['action_spaces'][i] for i, agent in enumerate(self.agents)
        }

        self.all_tasks = env_config['task_space']
        self.seed = env_config['seed']
        self.current_task = env_config.get('current_task', None)
        self.config = copy(env_config)
        # from policies.shared import logging
        # self.logging = logging

    def reset(self):
        self.env.reset()
        obs = self.env.get_obs()
        return {agent: obs[i] for i, agent in enumerate(self.agents)}

    def step(self, action):
        action = [action[agent] for agent in self.agents]
        reward, done, _ = self.env.step(action)
        obs = self.env.get_obs()

        assert self.config['reward_type'] == 0
        if self.config['reward_type'] == 0:
            reward = 1.0
        elif self.config['reward_type'] == 1:
            reward = 1.0 if reward > self.RANGE else 0.0
        else:
            reward = reward

        # Recalculate reward based on current_task
        if self.current_task is None:
            # raise ValueError("No task selected")
            ret = [{agent: obs[i] for i, agent in enumerate(self.agents)},
                   {agent: 0 for i, agent in enumerate(self.agents)},
                   {agent: done for i, agent in enumerate(self.agents)},
                   {agent: None for i, agent in enumerate(self.agents)}]
            return ret

        state = self.env.get_state()
        angles = state[1:3]
        # check if two angles meet the task
        correct = []
        for i in range(0, len(angles)):
            correct.append(abs(angles[i] - self.current_task) < self.EPSILON)
        if sum(correct) == 2:
            reward = 1
            # self.logging.info("YAY!")
        elif sum(correct) == 0:
            reward *= 0.5
        else:
            reward *= 0

        ret = [{agent: obs[i] for i, agent in enumerate(self.agents)},
               {agent: reward for i, agent in enumerate(self.agents)},
               {agent: done for i, agent in enumerate(self.agents)},
               {agent: None for i, agent in enumerate(self.agents)}]
        return ret

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def max_num_agents(self):
        return len(self.possible_agents)

    def __str__(self):
        return "Multi_Swimmer_v2"

    @property
    def unwrapped(self):
        return self.env

    def render(self, mode="human"):
        return [self.env.env.render(mode=mode)]
"""