from pycigar.envs.multiagent import MultiEnv
from pycigar.envs.wrappers.action_wrappers import *
from pycigar.envs.wrappers.observation_wrappers import *
from pycigar.envs.wrappers.reward_wrappers import *
from pycigar.envs.wrappers.wrapper import Wrapper


class AdvEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)  # receive a dict of rl_id: action
        env = AllRelativeInitDiscreteActionWrapper(env)
        env = AdvObservationWrapper(env)
        env = AdvLocalRewardWrapper(env)
        env = GroupActionWrapper(env)  # grouping layer
        env = GroupRewardWrapper(env)
        env = GroupObservationWrapper(env)  # grouping layer
        env = GroupInfoWrapper(env)
        env = AdvFramestackObservationWrapper(env)
        self.env = env

class ClusterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = SingleRelativeInitPhaseSpecificDiscreteActionWrapper(env)
        env = ClusterObservationWrapper(env)
        env = ClusterFramestackObservationWrapper(env)
        env = ClusterRewardWrapper(env)
        self.env = env

class ClusterMultiEnv(MultiEnv):
    def __init__(self, **kwargs):
        self.env = ClusterEnv(**kwargs)
        self.env.action_space
        self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action_dict):
        return self.env.step(action_dict)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

class GroupInfoWrapper(Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, rl_actions, randomize_rl_update=None):
        observation, reward, done, info = self.env.step(rl_actions, randomize_rl_update)
        return observation, reward, done, self.info(info)

    def info(self, info):
        new_info = {}
        new_info['defense_agent'] = {key: info[key] for key in info if 'adversary_' not in key}
        new_info['attack_agent'] = {key: info[key] for key in info if 'adversary_' in key}
        if not new_info['defense_agent']:
            del new_info['defense_agent']
        if not new_info['attack_agent']:
            del new_info['attack_agent']

        return new_info


class AdvMultiEnv(MultiEnv):
    def __init__(self, **kwargs):
        self.env = AdvEnv(**kwargs)
        self.env.action_space
        self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action_dict):
        return self.env.step(action_dict)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)



class PhaseSpecificContinuousMultiEnv(MultiEnv):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = RelativeInitContinuousActionWrapper(env)
        env = AdvObservationWrapper(env, unbalance=True)
        env = AdvFramestackObservationWrapper(env)
        env = PhaseSpecificRewardWrapper(env, unbalance=True)
        # env = CentralGlobalRewardWrapper(env, unbalance=True)
        # env = CentralLocalObservationWrapper(env, unbalance=True)
        # env = CentralFramestackObservationWrapper(env)
        self.env = env
        self.env.action_space
        self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action_dict):
        return self.env.step(action_dict)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

