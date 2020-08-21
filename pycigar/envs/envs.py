from pycigar.envs.central_env import CentralEnv
from pycigar.envs.wrappers import *


class CentralControlPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = SingleRelativeInitDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env


class NewCentralControlPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = NewSingleRelativeInitDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env


class CentralControlPVInverterContinuousEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = SingleRelativeInitContinuousActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env


class CentralControlPhaseSpecificPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = SingleRelativeInitPhaseSpecificDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env, unbalance=True)
        env = CentralLocalObservationWrapper(env, unbalance=True)
        env = CentralFramestackObservationWrapper(env)
        env = CentralLocalPhaseSpecificObservationWrapper(env, unbalance=True)
        self.env = env


class CentralControlPhaseSpecificContinuousPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = SingleRelativeInitPhaseSpecificContinuousActionWrapper(env)
        env = CentralGlobalRewardWrapper(env, unbalance=True)
        env = CentralLocalObservationWrapper(env, unbalance=True)
        env = CentralFramestackObservationWrapper(env)
        self.env = env
