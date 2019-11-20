from pycigar.envs.multiagent import MultiEnv
from gym.spaces import Dict, Tuple, Discrete, Box
import numpy as np


class Wrapper(MultiEnv):

    def __init__(self, env):
        self.env = env

    def step(self, rl_actions):
        return self.env.step(rl_actions)

    def reset(self):
        return self.env.reset()

    def plot(self, exp_tag='', env_name='', iteration=0):
        return self.env.plot(exp_tag, env_name, iteration)

    def get_old_actions(self):
        return self.env.get_old_actions()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space


class ObservationWrapper(Wrapper):
    def reset(self):
        observation = self.env.reset()
        return self.observation(observation)

    def step(self, rl_actions):
        observation, reward, done, info = self.env.step(rl_actions)
        return self.observation(observation), reward, done, info

    @property
    def observation_space(self):
        return NotImplementedError

    def observation(self, observation):
        raise NotImplementedError


class ActionWrapper(Wrapper):

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        raise NotImplementedError


class RewardWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        raise NotImplementedError
