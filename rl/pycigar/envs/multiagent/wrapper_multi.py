from pycigar.envs.multiagent import MultiEnv
import numpy as np 

"""
The abstract definition of wrapper.
"""


class Wrapper(MultiEnv):

    def __init__(self, env):
        self.env = env
        self.INIT_ACTION = {}
        k = env.get_kernel()
        pv_device_ids = k.device.get_pv_device_ids()
        for device_id in pv_device_ids:
            self.INIT_ACTION[device_id] = np.array(k.device.get_control_setting(device_id))

    def step(self, rl_actions):
        return self.env.step(rl_actions)

    def reset(self):
        return self.env.reset()

    def plot(self, exp_tag='', env_name='', iteration=0, reward=0):
        return self.env.plot(exp_tag, env_name, iteration, reward)

    def get_kernel(self):
        return self.env.get_kernel()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def base_env(self):
        return self.env.base_env


class ObservationWrapper(Wrapper):
    def reset(self):
        observation = self.env.reset()
        return self.observation(observation, info=None)

    def step(self, rl_actions):
        observation, reward, done, info = self.env.step(rl_actions)
        return self.observation(observation, info), reward, done, info


    @property
    def observation_space(self):
        return NotImplementedError

    def observation(self, observation, info):
        raise NotImplementedError


class ActionWrapper(Wrapper):

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action, info=None):
        raise NotImplementedError


class RewardWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, info), done, info

    def reward(self, reward, info):
        raise NotImplementedError



from pycigar.envs import CentralEnv
import numpy as np 

"""
The abstract definition of wrapper.
"""


class CentralWrapper(CentralEnv):

    def __init__(self, env):
        self.env = env
        k = env.get_kernel()

    def step(self, rl_actions):
        return self.env.step(rl_actions)

    def reset(self):
        observation = self.env.reset()
        self.INIT_ACTION = self.env.INIT_ACTION
        return observation 
    
    def plot(self, exp_tag='', env_name='', iteration=0, reward=0):
        return self.env.plot(exp_tag, env_name, iteration, reward)

    def get_pycigar_output_specs(self):
        return self.env.get_pycigar_output_specs()
    def get_kernel(self):
        return self.env.get_kernel()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def base_env(self):
        return self.env.base_env


class CentralObservationWrapper(CentralWrapper):
    def reset(self):
        observation = self.env.reset()
        self.INIT_ACTION = self.env.INIT_ACTION
        return self.observation(observation, info=None)

    def step(self, rl_actions):
        observation, reward, done, info = self.env.step(rl_actions)
        return self.observation(observation, info), reward, done, info


    @property
    def observation_space(self):
        return NotImplementedError

    def observation(self, observation, info):
        raise NotImplementedError


class CentralActionWrapper(CentralWrapper):

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action, info=None):
        raise NotImplementedError


class CentralRewardWrapper(CentralWrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, info), done, info

    def reward(self, reward, info):
        raise NotImplementedError
