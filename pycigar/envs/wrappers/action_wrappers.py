from gym.spaces import Tuple, Discrete, Box
from ray.tune.utils import merge_dicts

from pycigar.envs.wrappers.wrapper import Wrapper
from pycigar.envs.wrappers.wrappers_constants import *
from pycigar.utils.logging import logger

class ActionWrapper(Wrapper):
    def step(self, action, randomize_rl_update=None):
        rl_actions = {}
        info_update = {}
        if isinstance(action, dict):
            # multi-agent env
            for i, a in action.items():
                rl_actions[i] = self.action(a, i)
                info_update[i] = {'raw_action': a}
        else:
            # central env
            for i in self.k.device.get_rl_device_ids():
                rl_actions[i] = self.action(action, i)
                info_update[i] = {'raw_action': action}

        observation, reward, done, info = self.env.step(rl_actions, randomize_rl_update)

        if info:
            info_update_keys = list(info_update.keys())
            for key in info_update_keys:
                if key not in info.keys():
                    del info_update[key]

        return observation, reward, done, merge_dicts(info, info_update)

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

    def action(self, action, rl_id, *_):
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

    def action(self, action, rl_id, *_):
        Logger = logger()
        Logger.log('component_action', rl_id, self.INIT_ACTION[rl_id] - ACTION_RANGE + ACTION_STEP * action)
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
        return Tuple([Discrete(DISCRETIZE_RELATIVE)] * 2)

    def action(self, action, rl_id, *_):
        act = self.INIT_ACTION[rl_id] - ACTION_RANGE + ACTION_STEP * action[0]
        act[0] = act[1] - (ACTION_MAX_SLOPE - ACTION_MIN_SLOPE) / DISCRETIZE_RELATIVE * action[1] - ACTION_MIN_SLOPE
        act[3] = act[1] + (ACTION_MAX_SLOPE - ACTION_MIN_SLOPE) / DISCRETIZE_RELATIVE * action[1] + ACTION_MIN_SLOPE
        return act


class AllRelativeInitDiscreteActionWrapper(ActionWrapper):
    """
    Action head is only 1 value.
    The action head is 1 action discretized into DISCRETIZE_RELATIVE number of bins.
    We control 5 VBPs by translate the VBPs.
    Each bin is a step of ACTION_STEP deviated from the initial action.
    """

    @property
    def action_space(self):
        return Tuple([Discrete(DISCRETIZE_RELATIVE)] * 5)

    def action(self, action, rl_id, *_):
        act = self.INIT_ACTION[rl_id] - ACTION_RANGE + ACTION_STEP * action[2]
        act[1] = act[2] - (ACTION_MAX_SLOPE - ACTION_MIN_SLOPE) / DISCRETIZE_RELATIVE * action[1] - ACTION_MIN_SLOPE
        act[0] = act[1] - (ACTION_MAX_SLOPE - ACTION_MIN_SLOPE) / DISCRETIZE_RELATIVE * action[0] - ACTION_MIN_SLOPE
        act[3] = act[2] + (ACTION_MAX_SLOPE - ACTION_MIN_SLOPE) / DISCRETIZE_RELATIVE * action[3] + ACTION_MIN_SLOPE
        act[4] = act[3] + (ACTION_MAX_SLOPE - ACTION_MIN_SLOPE) / DISCRETIZE_RELATIVE * action[4] + ACTION_MIN_SLOPE
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
        return Box(-1.0, 1.0, (1,), dtype=np.float64)

    def action(self, action, rl_id, *_):
        return self.INIT_ACTION[rl_id] + action


#########################
#    UNBALANCE          #
#########################

class SingleRelativeInitPhaseSpecificDiscreteActionWrapper(ActionWrapper):
    """
    Action head is 4 values:
        - one for VBP translation for inverters on phase a
        - one for VBP translation for inverters on phase b
        - one for VBP translation for inverters on phase b
        - one for VBP translation for inverters on three phases
    """

    @property
    def action_space(self):
        return Tuple([Discrete(DISCRETIZE_RELATIVE)] * 3)

    def action(self, action, rl_id, *_):
        if rl_id.endswith('a'):
            translation = action[0]
        elif rl_id.endswith('b'):
            translation = action[1]
        elif rl_id.endswith('c'):
            translation = action[2]
        else:
            translation = int(DISCRETIZE_RELATIVE / 2)

        return self.INIT_ACTION[rl_id] - ACTION_RANGE + ACTION_STEP * translation


class SingleRelativeInitPhaseSpecificContinuousActionWrapper(ActionWrapper):
    """
    Action head is 4 values:
        - one for VBP translation for inverters on phase a
        - one for VBP translation for inverters on phase b
        - one for VBP translation for inverters on phase b
        - one for VBP translation for inverters on three phases
    """

    @property
    def action_space(self):
        return Box(-10.0, 10.0, (3,), dtype=np.float64)

    def action(self, action, rl_id, *_):
        action = np.array(action) * ACTION_RANGE / 10.0
        if rl_id.endswith('a'):
            translation = action[0]
        elif rl_id.endswith('b'):
            translation = action[1]
        elif rl_id.endswith('c'):
            translation = action[2]
        else:
            translation = 0

        return self.INIT_ACTION[rl_id] + translation

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

    def action(self, action, rl_id, *_):
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


#############################
#    MULTI-AGENT WRAPPER    #
#############################

class GroupActionWrapper(Wrapper):
    def step(self, action, randomize_rl_update=None):
        rl_actions = {}
        if isinstance(action, dict):
            # multi-agent env
            if 'defense_agent' in action:
                rl_actions = {device_name: action['defense_agent'] for device_name in self.k.device.get_rl_device_ids() if 'adversary_' not in device_name}
            if 'attack_agent' in action:
                rl_actions = {device_name: action['attack_agent'] for device_name in self.k.device.get_rl_device_ids() if 'adversary_' in device_name}

        observation, reward, done, info = self.env.step(rl_actions, randomize_rl_update)
        return observation, reward, done, info
