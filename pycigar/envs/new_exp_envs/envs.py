from env import CentralEnv
from wrapper import DelayObservation, CentralFramestackObservationWrapper
from pycigar.envs.wrappers.action_wrappers import SingleRelativeInitPhaseSpecificDiscreteActionWrapper
from pycigar.envs.wrappers.reward_wrappers import CentralGlobalRewardWrapper
from pycigar.envs.wrappers.wrapper import Wrapper

class NewExpCentralControlPhaseSpecificPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = SingleRelativeInitPhaseSpecificDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env, multi_attack=True)
        env = DelayObservation(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env
