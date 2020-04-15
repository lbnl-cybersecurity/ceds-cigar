from pycigar.envs.multiagent import MultiEnv
from pycigar.envs.wrappers.action_wrappers import *
from pycigar.envs.wrappers.observation_wrappers import *
from pycigar.envs.wrappers.reward_wrappers import *
from pycigar.envs.wrappers.wrapper import Wrapper


class ARDiscreteIACEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = LocalRewardWrapper(env)
        env = ARDiscreteActionWrapper(env)
        env = LocalObservationWrapper(env)
        self.env = env


class SingleDiscreteIACEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = LocalRewardWrapper(env)
        env = SingleDiscreteActionWrapper(env)
        env = LocalObservationWrapper(env)
        self.env = env


class ARDiscreteCoopEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = GlobalRewardWrapper(env)
        env = ARDiscreteActionWrapper(env)
        env = LocalObservationWrapper(env)
        self.env = env


class SingleDiscreteCoopEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = GlobalRewardWrapper(env)
        env = SingleDiscreteActionWrapper(env)
        env = LocalObservationWrapper(env)
        self.env = env


class FramestackSingleDiscreteCoopEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = GlobalRewardWrapper(env)
        env = SingleDiscreteActionWrapper(env)
        env = LocalObservationWrapper(env)
        env = FramestackObservationWrapper(env)
        self.env = env


class SingleRelativeDiscreteCoopEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = SecondStageGlobalRewardWrapper(env)
        env = SingleRelativeInitDiscreteActionWrapper(env)
        env = LocalObservationV4Wrapper(env)
        env = FramestackObservationV4Wrapper(env)
        env = GlobalObservationWrapper(env)
        self.env = env


# Coma environment
# Not in use
class ARDiscreteComaEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = LocalRewardWrapper(env)
        env = ARDiscreteActionWrapper(env)
        env = GlobalObservationWrapper(env)
        self.env = env


class SingleDiscreteComaEnv(Wrapper):
    def __init__(self, **kwargs):
        env = MultiEnv(**kwargs)
        env = GlobalRewardWrapper(env)
        env = SingleDiscreteActionWrapper(env)
        env = GlobalObservationWrapper(env)
        self.env = env
