from pycigar.envs.rl_control_pv_inverter_env import RLControlPVInverterEnv
from pycigar.envs.wrappers import *


class CentralControlPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = RLControlPVInverterEnv(**kwargs)
        env = SingleRelativeInitDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env


class NewCentralControlPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = RLControlPVInverterEnv(**kwargs)
        env = NewSingleRelativeInitDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env


class CentralControlPVInverterContinuousEnv(Wrapper):
    def __init__(self, **kwargs):
        env = RLControlPVInverterEnv(**kwargs)
        env = SingleRelativeInitContinuousActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env

class CentralControlPhaseSpecificPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = RLControlPVInverterEnv(**kwargs)
        env = SingleRelativeInitPhaseSpecificDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env
