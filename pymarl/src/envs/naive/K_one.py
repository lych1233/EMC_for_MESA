import pickle
import gym
import numpy as np
from gym import spaces
import random


# Fixed number of agents
# Task: k people need to perform action 0 to get the highest reward
# Episode length: 1 (matrix game)


class K_one(gym.Env):
    def __init__(self, init_seed):
        super(K_one, self).__init__()
        self.n_agent = 2
        self.single_obs_dim = 1
        self._n_act = 10

        rng = np.random.RandomState(init_seed)
        self.goal = rng.randint(0, 10)

        self.observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(self.single_obs_dim,), dtype=np.float32)] * self.n_agent
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(self.n_agent * self.single_obs_dim,), dtype=np.float32)] * self.n_agent
        self.action_space = [spaces.Discrete(self._n_act)] * self.n_agent

    def get_current_obs(self):
        return [np.array([0.]) for _ in range(self.n_agent)]

    def render(self, mode="human"):
        pass

    def reset(self):
        return self.get_current_obs()

    def step(self, action):
        cnt = 0
        for a in action:
            if a[0] == self.goal:
                cnt += 1
        if cnt == 1:
            reward = 1
        elif cnt == 0:
            reward = 0.5
        else:
            reward = 0

        done = True
        return self.get_current_obs(), [[reward]] * self.n_agent, [done] * self.n_agent, [{}] * self.n_agent
    
    def seed(self, seed=None):
        pass


if __name__ == '__main__':
    env = ClimbGame_Single_10()
    obs = env.reset()
    for i in range(3):
        obs, reward, done, info = env.step([[1], [2]])
        print(obs, reward, done, info)
