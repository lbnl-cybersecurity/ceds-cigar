from pycigar.envs.multiagent import MultiEnv
from pycigar.envs.wrappers.action_wrappers import *
from pycigar.envs.wrappers.observation_wrappers import *
from pycigar.envs.wrappers.reward_wrappers import *
from pycigar.envs.wrappers.wrapper import Wrapper


class AdvMultiEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)                           # receive a dict of rl_id: action
        env = AllRelativeInitDiscreteActionWrapper(env)
        env = AdvObservationWrapper(env)
        env = LocalRewardWrapper(env)
        env = GroupActionWrapper(env)                      # grouping layer
        env = GroupObservationWrapper(env)                 # grouping layer
        env = GroupRewardWrapper(env)
        self.env = env
