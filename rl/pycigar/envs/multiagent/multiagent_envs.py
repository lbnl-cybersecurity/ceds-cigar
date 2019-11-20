from pycigar.envs.multiagent.wrappers import LocalObservationWrapper
from pycigar.envs.multiagent.wrappers import LocalObservationV2Wrapper
from pycigar.envs.multiagent.wrappers import GlobalObservationWrapper
from pycigar.envs.multiagent.wrappers import FramestackObservationWrapper
from pycigar.envs.multiagent.wrappers import FramestackObservationV2Wrapper

from pycigar.envs.multiagent.wrappers import LocalRewardWrapper
from pycigar.envs.multiagent.wrappers import GlobalRewardWrapper

from pycigar.envs.multiagent.wrappers import SingleDiscreteActionWrapper
from pycigar.envs.multiagent.wrappers import SingleRelativeInitDiscreteActionWrapper
from pycigar.envs.multiagent.wrappers import SecondStageGlobalRewardWrapper
# from pycigar.envs.multiagent.wrappers import SingleContinuousActionWrapper
from pycigar.envs.multiagent.wrappers import ARDiscreteActionWrapper
# from pycigar.envs.multiagent.wrappers import ARContinuousActionWrapper
from pycigar.envs.multiagent import MultiEnv
from pycigar.envs.multiagent.wrapper_multi import Wrapper


# IAC environment
class ARDiscreteIACEnv(Wrapper):

    """Environment for autoregressive action head, Independent Actor-Critic experiment.
    """

    def __init__(self, **kwargs):
        self.env = LocalObservationWrapper(ARDiscreteActionWrapper(LocalRewardWrapper(MultiEnv(**kwargs))))


class SingleDiscreteIACEnv(Wrapper):

    """Environment for single action head, Independent Actor-Critic experiment.
    """
    def __init__(self, **kwargs):
        self.env = LocalObservationWrapper(SingleDiscreteActionWrapper(LocalRewardWrapper(MultiEnv(**kwargs))))


# Cooperative environment
class ARDiscreteCoopEnv(Wrapper):

    """Environment for autoregressive action head, cooperative strategy experiment (reward is the average of all rewards).
    """
    def __init__(self, **kwargs):
        self.env = LocalObservationWrapper(ARDiscreteActionWrapper(GlobalRewardWrapper(MultiEnv(**kwargs))))


class SingleDiscreteCoopEnv(Wrapper):

    """Environment for single action head, cooperative strategy experiment (reward is the average of all rewards).
    Translate the VBPs.
    """
    def __init__(self, **kwargs):
        self.env = FramestackObservationV2Wrapper(LocalObservationV2Wrapper(SingleDiscreteActionWrapper(GlobalRewardWrapper(MultiEnv(**kwargs)))))


class SecondStageSingleDiscreteCoopEnv(Wrapper):
    """Environment for single action head, cooperative strategy experiment (reward is the average of all rewards).
    Reward change to a complicated reward, use in curriculum learning.
    """
    def __init__(self, **kwargs):
        self.env = FramestackObservationV2Wrapper(LocalObservationV2Wrapper(SingleDiscreteActionWrapper(SecondStageGlobalRewardWrapper(MultiEnv(**kwargs)))))


class SingleRelativeDiscreteCoopEnv(Wrapper):
    """Environment for single action head, cooperative strategy experiment (reward is the average of all rewards).
    Action is the deviation of VBPs to the initial VBPs.
    """
    def __init__(self, **kwargs):
        self.env = LocalObservationWrapper(SingleRelativeInitDiscreteActionWrapper(GlobalRewardWrapper(MultiEnv(**kwargs))))


class FramestackSingleDiscreteCoopEnv(Wrapper):
    """Environment for single action head, cooperative strategy experiment (reward is the average of all rewards).
    The observation is stacked.
    """
    def __init__(self, **kwargs):
        self.env = FramestackObservationWrapper(LocalObservationWrapper(SingleDiscreteActionWrapper(GlobalRewardWrapper(MultiEnv(**kwargs)))))

# Coma environment
class ARDiscreteComaEnv(Wrapper):
    def __init__(self, **kwargs):
        self.env = GlobalObservationWrapper(ARDiscreteActionWrapper(LocalRewardWrapper(MultiEnv(**kwargs))))


class SingleDiscreteComaEnv(Wrapper):
    def __init__(self, **kwargs):
        self.env = GlobalObservationWrapper(SingleDiscreteActionWrapper(GlobalRewardWrapper(MultiEnv(**kwargs))))
