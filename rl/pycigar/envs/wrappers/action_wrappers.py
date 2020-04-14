from gym.spaces import Tuple, Discrete, Box, MultiDiscrete

from pycigar.envs.wrappers.wrapper import Wrapper
from pycigar.envs.wrappers.wrappers_constants import *


class ActionWrapper(Wrapper):
    def step(self, action):
        rl_actions = {}
        if isinstance(action, dict):
            # multi-agent env
            for i, a in action.items():
                rl_actions[i] = self.action(a, i)
        else:
            # central env
            for i in self.INIT_ACTION:
                rl_actions[i] = self.action(action, i)

        return self.env.step(rl_actions)

    def action(self, action, rl_id):
        """Modify action before feed into the simulation.

        Parameters
        ----------
        action
            The action value we received from RLlib. Can be an integer or an array depending on the action space.

        Returns
        -------
        dict
            Action value with a valid form to feed into the environment.
        """
        raise NotImplementedError


#########################
#         SINGLE        #
#########################

class SingleDiscreteActionWrapper(ActionWrapper):
    """
    Action head is only 1 value.
    The action head is 1 action discretized into DISCRETIZE number of bins.
    We control 5 VBPs by translate the VBPs.
    The action we feed into the environment is ranging from ACTION_LOWER_BOUND->ACTION_UPPER_BOUND.
    """

    @property
    def action_space(self):
        return Discrete(DISCRETIZE)

    def action(self, action, rl_id):
        t = ACTION_LOWER_BOUND + (ACTION_UPPER_BOUND - ACTION_LOWER_BOUND) / DISCRETIZE * action
        return ACTION_CURVE + t


class SingleRelativeInitDiscreteActionWrapper(ActionWrapper):
    """
    Action head is only 1 value.
    The action head is 1 action discretized into DISCRETIZE_RELATIVE number of bins.
    We control 5 VBPs by translate the VBPs.
    Each bin is a step of ACTION_STEP deviated from the initial action.
    """

    @property
    def action_space(self):
        return Discrete(DISCRETIZE_RELATIVE)

    def action(self, action, rl_id):
        return self.INIT_ACTION[rl_id] - ACTION_RANGE + ACTION_STEP * action


# TODO: change name
class NewSingleRelativeInitDiscreteActionWrapper(ActionWrapper):
    """
    Action head is only 1 value.
    The action head is 1 action discretized into DISCRETIZE_RELATIVE number of bins.
    We control 5 VBPs by translate the VBPs.
    Each bin is a step of ACTION_STEP deviated from the initial action.
    """

    @property
    def action_space(self):
        return Discrete(DISCRETIZE_RELATIVE * DISCRETIZE_RELATIVE)

    def action(self, action, rl_id):
        rl_actions = ACTION_MAP[action]
        act = self.INIT_ACTION[rl_id] - ACTION_RANGE + ACTION_STEP * rl_actions[0]
        act[0] = act[1] - (ACTION_MAX_SLOPE - ACTION_MIN_SLOPE) / DISCRETIZE_RELATIVE * rl_actions[1] - ACTION_MIN_SLOPE
        act[3] = act[1] + (ACTION_MAX_SLOPE - ACTION_MIN_SLOPE) / DISCRETIZE_RELATIVE * rl_actions[1] + ACTION_MIN_SLOPE
        return act


class SingleRelativeInitContinuousActionWrapper(ActionWrapper):
    """
    Action head is only 1 value.
    The action head is 1 action discretized into DISCRETIZE_RELATIVE number of bins.
    We control 5 VBPs by translate the VBPs.
    Each bin is a step of ACTION_STEP deviated from the initial action.
    """

    @property
    def action_space(self):
        return Box(-1.0, 1.0, (1,), dtype=np.float32)

    def action(self, action, rl_id):
        return self.INIT_ACTION[rl_id] + action



#########################
#    AUTO REGRESSIVE    #
#########################

class ARDiscreteActionWrapper(ActionWrapper):
    """
    Action head is an array of 5 value.
    The action head is 5 action discretized into DISCRETIZE number of bins.
    We control all 5 breakpoints of inverters.
    """

    @property
    def action_space(self):
        return Tuple([Discrete(DISCRETIZE), Discrete(DISCRETIZE),
                      Discrete(DISCRETIZE), Discrete(DISCRETIZE),
                      Discrete(DISCRETIZE)])

    def action(self, action, rl_id):
        # This is used to form the discretized value into the valid action before feed into the environment.
        act = ACTION_LOWER_BOUND + (ACTION_UPPER_BOUND - ACTION_LOWER_BOUND) / DISCRETIZE * np.array(action, np.float32)
        # if the action returned by the agent violate the constraint (the next point is >= the current point),
        # then we apply a hard threshold on the next point.
        if act[1] < act[0]:
            act[1] = act[0]
        if act[2] < act[1]:
            act[2] = act[1]
        if act[3] < act[2]:
            act[3] = act[2]
        if act[4] < act[3]:
            act[4] = act[3]
        return act


class ARContinuousActionWrapper(ActionWrapper):
    pass
