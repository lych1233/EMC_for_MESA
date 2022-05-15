import pickle
import gym
import numpy as np
from gym import spaces
import random


# Fixed number of agents
# Task: k people need to perform action 0 to get the highest reward
# Episode length: 1 (matrix game)


class ClimbGame_Multi_X(gym.Env):
    def __init__(self, init_seed, _n_act=3):
        super(ClimbGame_Multi_X, self).__init__()
        self.n_agent = 2
        self.timestep = 5
        self._n_act = _n_act

        self.single_obs_dim = self.timestep

        rng = np.random.RandomState(init_seed)
        self.goal = [rng.randint(0, self._n_act) for _ in range(self.timestep)]

        self.observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(self.single_obs_dim,), dtype=np.float32)] * self.n_agent
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(self.n_agent * self.single_obs_dim,), dtype=np.float32)] * self.n_agent
        self.action_space = [spaces.Discrete(self._n_act)] * self.n_agent
        self._cur_step = 0.

    def get_current_obs(self):
        if self._cur_step >= self.timestep:
            return [np.zeros(self.single_obs_dim) for _ in range(self.n_agent)]
        return [np.eye(self.single_obs_dim)[int(self._cur_step)] for _ in range(self.n_agent)]

    def render(self, mode="human"):
        pass

    def reset(self):
        self._cur_step = 0.
        return self.get_current_obs()

    def step(self, action):
        cnt = 0
        for a in action:
            if a[0] == self.goal[int(self._cur_step)]:
                cnt += 1
        if cnt == 2:
            reward = 1.
        elif cnt == 0:
            reward = 0.5
        else:
            reward = 0.

        self._cur_step += 1
        if self._cur_step >= self.timestep:
            done = True
        else:
            done = False
        return self.get_current_obs(), [[reward]] * self.n_agent, [done] * self.n_agent, [{}] * self.n_agent

    def seed(self, seed=None):
        pass


if __name__ == '__main__':
    env = ClimbGame_Multi_3(init_seed=3)
    obs = env.reset()
    for i in range(6):
        obs, reward, done, info = env.step([[1], [2]])
        if done[0]:
            break
