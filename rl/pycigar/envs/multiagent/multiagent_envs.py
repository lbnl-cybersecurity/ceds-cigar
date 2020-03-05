from pycigar.envs.multiagent.wrappers import LocalObservationWrapper
from pycigar.envs.multiagent.wrappers import LocalObservationV2Wrapper
from pycigar.envs.multiagent.wrappers import LocalObservationV3Wrapper
from pycigar.envs.multiagent.wrappers import LocalObservationV4Wrapper

from pycigar.envs.multiagent.wrappers import GlobalObservationWrapper

from pycigar.envs.multiagent.wrappers import FramestackObservationWrapper
from pycigar.envs.multiagent.wrappers import FramestackObservationV2Wrapper
from pycigar.envs.multiagent.wrappers import FramestackObservationV3Wrapper
from pycigar.envs.multiagent.wrappers import FramestackObservationV4Wrapper

from pycigar.envs.multiagent.wrappers import LocalRewardWrapper
from pycigar.envs.multiagent.wrappers import GlobalRewardWrapper
from pycigar.envs.multiagent.wrappers import SearchGlobalRewardWrapper

from pycigar.envs.multiagent.wrappers import SingleDiscreteActionWrapper
from pycigar.envs.multiagent.wrappers import SingleRelativeInitDiscreteActionWrapper
from pycigar.envs.multiagent.wrappers import SecondStageGlobalRewardWrapper
# from pycigar.envs.multiagent.wrappers import SingleContinuousActionWrapper
from pycigar.envs.multiagent.wrappers import ARDiscreteActionWrapper
# from pycigar.envs.multiagent.wrappers import ARContinuousActionWrapper
from pycigar.envs.multiagent import MultiEnv
from pycigar.envs.multiagent.wrapper_multi import Wrapper

from pycigar.envs.multiagent.wrappers import CentralFramestackObservationWrapper
from pycigar.envs.multiagent.wrappers import CentralLocalObservationWrapper
from pycigar.envs.rl_control_pv_inverter_env import RLControlPVInverterEnv
from pycigar.envs.multiagent.wrappers import CentralSingleRelativeInitDiscreteActionWrapper
from pycigar.envs.multiagent.wrappers import CentralGlobalRewardWrapper
from pycigar.envs.multiagent.wrapper_multi import CentralWrapper
"""
Tutorial to add new environment:
    - Import your custom wrappers from pycigar/envs/multiagent/wrappers.py
    - Create the environment as same as the following examples below.
    - Go to pycigar/envs/multiagent/__init__.py,
       import your new environment and add the envinroment string name into the list.
    - Now you are able to run the experiment as the example in /examples/.
"""

###########################################################################
#             Independent Actor-Critic (IAC) environment                  #
###########################################################################
"""
Environment for Indepedent Actor-Critic experiment. In this experiment, each
agent will receive its local observation (without knowing other agents' observations and actions)
and the reward is the local reward (the reward of the agent is the reward at the devices, and it is
independent of other agents' rewards).
We have different environments for different action head experiments.
"""


class ARDiscreteIACEnv(Wrapper):

    """Observation: local observation
       Reward: local reward
       Action: auto-regressive action head (control all breakpoints,
                                            sampling next breakpoint conditional on previous breakpoints)
               discretize each breakpoint value into n bins.

       For further information, please read the description of wrappers in pycigar/envs/multiagent/wrapper_wrappers.py
    Attributes
    ----------
    env : Wrapper
        The environment object.
    """

    def __init__(self, **kwargs):
        self.env = LocalObservationWrapper(ARDiscreteActionWrapper(LocalRewardWrapper(MultiEnv(**kwargs))))


class SingleDiscreteIACEnv(Wrapper):

    """Observation: local observation
       Reward: local reward
       Action: control only one breakpoint (other breakpoints are functions of controlled breakpoints).
               for instance, other breakpoints translate with the controlled breakpoint.
               discretize breakpoint value into n bins.

       For further information, please read the description of wrappers in pycigar/envs/multiagent/wrapper_wrappers.py
    Attributes
    ----------
    env : Wrapper
        The environment object.
    """
    def __init__(self, **kwargs):
        self.env = LocalObservationWrapper(SingleDiscreteActionWrapper(LocalRewardWrapper(MultiEnv(**kwargs))))


###########################################################################
#                          Cooperative environment                        #
###########################################################################
"""
Environment for cooperative experiment. In this experiment, each
agent will receive its local observation (without knowing other agents' observations and actions).
and the reward is the global reward (the reward of the agent is the average reward of rewards received by all agents).
The reward could be changed for different experiment goals.
We have different environments for different action head experiments.
"""


class ARDiscreteCoopEnv(Wrapper):

    """Observation: local observation
       Reward: global reward
       Action: auto-regressive action head (control all breakpoints,
                                            sampling next breakpoint conditional on previous breakpoints)
               discretize each breakpoint value into n bins.

       For further information, please read the description of wrappers in pycigar/envs/multiagent/wrapper_wrappers.py

    Attributes
    ----------
    env : Wrapper
        The environment object.
    """

    def __init__(self, **kwargs):
        self.env = LocalObservationWrapper(ARDiscreteActionWrapper(GlobalRewardWrapper(MultiEnv(**kwargs))))


class SingleDiscreteCoopEnv(Wrapper):

    """Observation: local observation
       Reward: global reward (change the reward function in GlobalRewardWrapper().
                              For now, the reward is to encourage the agent not to take other actions
                              which is not the initial action)
       Action: control only one breakpoint (other breakpoints are functions of controlled breakpoints).
               for instance, other breakpoints translate with the controlled breakpoint.
               discretize breakpoint value into n bins.

       For further information, please read the description of wrappers in pycigar/envs/multiagent/wrapper_wrappers.py
    Attributes
    ----------
    env : Wrapper
        The environment object.
    """
    def __init__(self, **kwargs):
        self.env = LocalObservationWrapper(SingleDiscreteActionWrapper(GlobalRewardWrapper(MultiEnv(**kwargs))))


class SecondStageSingleDiscreteCoopEnv(Wrapper):

    """Observation: local observation
       Reward: global reward (change the reward function in SecondStageGlobalRewardWrapper().
                              For now, the reward is much complicated, require the agent to damp the oscillation
                              and encourage it to move back to initial action under normal conditions).
       Action: control only one breakpoint (other breakpoints are functions of controlled breakpoints).
               for instance, other breakpoints translate with the controlled breakpoint.
               discretize breakpoint value into n bins.

       The SingleDiscreteCoopEnv() and SecondStageSingleDiscreteCoopEnv() are used for curriculum learning.
       In curriculum learning, we train the agent on simple tasks and incrementally changing the difficulty of the tasks
       by changing the reward function.
       In this experiment, in first step, we encourage the agent to prefer to take initial action under normal conditions.
                           in second step, we want the agent to reduce the grid oscillation and take the initial action under
                           normal conditions.
       For further information, please read the description of wrappers in pycigar/envs/multiagent/wrapper_wrappers.py
    Attributes
    ----------
    env : Wrapper
        The environment object.
    """
    def __init__(self, **kwargs):
        self.env = SingleDiscreteActionWrapper(SecondStageGlobalRewardWrapper(MultiEnv(**kwargs)))


class FramestackSingleDiscreteCoopEnv(Wrapper):

    """Observation: stack of local observations in n timesteps.
       Reward: global reward (change the reward function in GlobalRewardWrapper().
                              For now, the reward is to encourage the agent not to take other actions
                              which is not the initial action)
       Action: control only one breakpoint (other breakpoints are functions of controlled breakpoints).
               for instance, other breakpoints translate with the controlled breakpoint.
               discretize breakpoint value into n bins.

       For further information, please read the description of wrappers in pycigar/envs/multiagent/wrapper_wrappers.py
    Attributes
    ----------
    env : Wrapper
        The environment object.
    """
    def __init__(self, **kwargs):
        self.env = FramestackObservationWrapper(LocalObservationWrapper(SingleDiscreteActionWrapper(GlobalRewardWrapper(MultiEnv(**kwargs)))))


class SingleRelativeDiscreteCoopEnv(Wrapper):
    """Observation: local observation
       Reward: global reward (change the reward function in GlobalRewardWrapper().
                              For now, the reward is to encourage the agent not to take other actions
                              which is not the initial action)
       Action: control only one breakpoint (other breakpoints are functions of controlled breakpoints).
               for instance, the controlled breakpoint is the relative distance between itself and the intial action.
               discretize breakpoint value into n bins.

       For further information, please read the description of wrappers in pycigar/envs/multiagent/wrapper_wrappers.py
    Attributes
    ----------
    env : Wrapper
        The environment object.
    """
    def __init__(self, **kwargs):
        self.env = GlobalObservationWrapper(FramestackObservationV4Wrapper(LocalObservationV4Wrapper(SingleRelativeInitDiscreteActionWrapper(SecondStageGlobalRewardWrapper(MultiEnv(**kwargs))))))


# Coma environment
# Not in use
class ARDiscreteComaEnv(Wrapper):
    def __init__(self, **kwargs):
        self.env = GlobalObservationWrapper(ARDiscreteActionWrapper(LocalRewardWrapper(MultiEnv(**kwargs))))


class SingleDiscreteComaEnv(Wrapper):
    def __init__(self, **kwargs):
        self.env = GlobalObservationWrapper(SingleDiscreteActionWrapper(GlobalRewardWrapper(MultiEnv(**kwargs))))


###########################################################
class CentralControlPVInverterEnv(CentralWrapper):
    def __init__(self, **kwargs):
        self.env = CentralFramestackObservationWrapper(CentralLocalObservationWrapper(CentralGlobalRewardWrapper(CentralSingleRelativeInitDiscreteActionWrapper(RLControlPVInverterEnv(**kwargs)))))
        #self.env = RLControlPVInverterEnv(**kwargs)
        
