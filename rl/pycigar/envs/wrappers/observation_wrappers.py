from collections import deque

from gym.spaces import Dict, Box, Tuple, Discrete

from pycigar.envs.wrappers.wrapper import Wrapper
from pycigar.envs.wrappers.wrappers_constants import *


class ObservationWrapper(Wrapper):
    def reset(self):
        observation = self.env.reset()
        self.INIT_ACTION = self.INIT_ACTION
        return self.observation(observation, info=None)

    def step(self, rl_actions, randomize_rl_update=None):
        observation, reward, done, info = self.env.step(rl_actions, randomize_rl_update)
        return self.observation(observation, info), reward, done, info

    @property
    def observation_space(self):
        return NotImplementedError

    def observation(self, observation, info):
        """
        Modifying the original observation into the observation that we want.

        Parameters
        ----------
        observation : dict
            The observation from the lastest wrapper.
        info : dict
            Additional information returned by the environment after one environment step.

        Returns
        -------
        dict
            new observation we want to feed into the RLlib.
        """
        raise NotImplementedError


###########################################################################
#                           CENTRAL WRAPPER                               #
###########################################################################

class CentralLocalObservationWrapper(ObservationWrapper):
    """ ATTENTION: this wrapper is only used with single head RELATIVE ACTION wrappers (control only 1 breakpoint).
    Observation: a dictionary of local observation for each agent, in the form of {'id_1': obs1, 'id_2': obs2,...}.
                 each agent observation is an array of [y-value, init_action_onehot, last_action_onehot]
                 at current timestep.
                y_value: the value measuring oscillation magnitude of the volage at the agent position.
                last_action_onehot: the last timestep action under one-hot encoding form.
                one-hot encoding: an array of zeros everywhere and have a value of 1 at the executed position.
                For example, we discretize the action space into DISCRETIZE=10 bins, and the action sent back from
                RLlib is: 3. The one-hot encoding of the action is: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    """

    def __init__(self, env, unbalance=False):
        super().__init__(env)
        self.unbalance = unbalance

    @property
    def observation_space(self):
        a_space = self.action_space
        if isinstance(a_space, Tuple):
            self.a_size = sum(a.n for a in a_space)
            # relative action, init is centered
            self.init_action = [int(a.n / 2) for a in a_space]
        elif isinstance(a_space, Discrete):
            self.a_size = a_space.n
            self.init_action = int(a_space.n / 2)  # action is discrete relative, so init is N / 2
        elif isinstance(a_space, Box):
            self.a_size = sum(a_space.shape)
            self.init_action = np.zeros(self.a_size)  # action is continuous relative, so init is 0
        else:
            raise NotImplementedError()

        return Box(low=-float('inf'), high=float('inf'), shape=(2 + self.a_size,), dtype=np.float64)

    def observation(self, observation, info):
        if info:
            old_actions = info[list(info.keys())[0]]['raw_action']
            p_set = np.mean([info[k]['p_set'] for k in self.k.device.get_rl_device_ids()])
        else:
            old_actions = self.init_action
            p_set = 0

        if isinstance(self.action_space, Tuple):
            # multihot
            old_a_encoded = np.zeros(self.a_size)
            offsets = np.cumsum([0, *[a.n for a in self.action_space][:-1]])
            for action, offset in zip(old_actions, offsets):
                old_a_encoded[offset + action] = 1

        elif isinstance(self.action_space, Discrete):
            # onehot
            old_a_encoded = np.zeros(self.a_size)
            old_a_encoded[old_actions] = 1
        elif isinstance(self.action_space, Box):
            old_a_encoded = old_actions.flatten()

        if self.unbalance:
            observation = np.array([observation['u'], p_set, *old_a_encoded])
        else:
            observation = np.array([observation['y'], p_set, *old_a_encoded])

        return observation


class CentralFramestackObservationWrapper(ObservationWrapper):
    """
    The return value of this wrapper is:
        [y_value_max, *previous_observation]
    Attributes
    ----------
    frames : deque
        A deque to keep the latest observations.
    """

    @property
    def observation_space(self):
        obss = self.env.observation_space
        if type(obss) is Box:
            self.frames = deque([], maxlen=NUM_FRAMES)
            shp = obss.shape
            obss = Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(shp[0] + 1,),
                dtype=np.float64)
        return obss

    def reset(self):
        # get the observation from environment as usual
        self.frames = deque([], maxlen=NUM_FRAMES)
        observation = self.env.reset()
        # add the observation into frames
        self.frames.append(observation)
        return self._get_ob()

    def observation(self, observation, info):
        # everytime we get a new observation from the environment, we add it to the frame and return the
        # post-processed observation.
        self.frames.append(observation)
        return self._get_ob()

    def _get_ob(self):
        y_value_max = max([obs[0] for obs in self.frames])
        new_obs = np.insert(self.frames[-1], 0, y_value_max)
        return new_obs


###########################################################################
#                         MULTI-AGENT WRAPPER                             #
###########################################################################
class GroupObservationWrapper(ObservationWrapper):
    def __init__(self, env, unbalance=False):
        super().__init__(env)
        self.unbalance = unbalance

    @property
    def observation_space(self):
        self.env.observation_space

    def observation(self, observation, info):
        obs = {}
        obs['defense_agent'] = np.mean([observation[key] for key in observation if 'adversary_' not in key])
        obs['attack_agent'] = np.mean([observation[key] for key in observation if 'adversary_' in key])
        if np.isnan(obs['defense_agent']):
            del obs['defense_agent']
        if np.isnan(obs['attack_agent']):
            del obs['attack_agent']

        return obs


class AdvObservationWrapper(ObservationWrapper):
    def __init__(self, env, unbalance=False):
        super().__init__(env)
        self.unbalance = unbalance

    @property
    def observation_space(self):
        a_space = self.action_space
        if isinstance(a_space, Tuple):
            self.a_size = sum(a.n for a in a_space)
            # relative action, init is centered
            self.init_action = [int(a.n / 2) for a in a_space]
        elif isinstance(a_space, Discrete):
            self.a_size = a_space.n
            self.init_action = int(a_space.n / 2)  # action is discrete relative, so init is N / 2
        elif isinstance(a_space, Box):
            self.a_size = sum(a_space.shape)
            self.init_action = np.zeros(self.a_size)  # action is continuous relative, so init is 0
        else:
            raise NotImplementedError()

        return Box(low=-float('inf'), high=float('inf'), shape=(2 + self.a_size,), dtype=np.float64)

    def observation(self, observation, info):

        if info:
            old_actions = {}
            p_set = {}
            for key in observation:
                # TODO: check for the else condition
                old_actions[key] = info[key]['raw_action'] if 'raw_action' in info[key] else self.init_action
                p_set[key] = info[key]['p_set'] if 'p_set' in info[key] else 0

        else:
            old_actions = {key: self.init_action for key in observation}
            p_set = {key: 0 for key in observation}

        if isinstance(self.action_space, Tuple):
            # multihot
            old_a_encoded = {}
            for key in old_actions:
                old_a_encoded[key] = np.zeros(self.a_size)
                offsets = np.cumsum([0, *[a.n for a in self.action_space][:-1]])
                for action, offset in zip(old_actions[key], offsets):
                    old_a_encoded[key][offset + action] = 1

        elif isinstance(self.action_space, Discrete):
            # onehot
            old_a_encoded = {}
            for key in old_actions:
                old_a_encoded[key] = np.zeros(self.a_size)
                old_a_encoded[key][old_actions[key]] = 1

        elif isinstance(self.action_space, Box):
            old_a_encoded = {}
            for key in old_actions:
                old_a_encoded[key] = old_actions[key].flatten()

        if self.unbalance:
            observation = {key: np.array([observation[key]['u'], p_set[key], *old_a_encoded[key]]) for key in observation}
        else:
            observation = {key: np.array([observation[key]['y'], p_set[key], *old_a_encoded[key]]) for key in observation}

        return observation


# for reference purpose
class GlobalObservationWrapper(ObservationWrapper):
    """
    Observation: a dictionary of global observation for each agent, in the form of:
                 {'id_1': {
                    'own_obs': local observation of the agent
                    'opponent_obs': concatenation of local observation of other agents
                    'opponent_action': concatenation of local observation of other agents (will be filled in at post-processing)
                 },
                 'id_2': {}
                 ,...}.

                 each agent observation is an array of [y-value, y-value-max, last_action_onehot, y-y_t5]
                 at current timestep.

                y_value: the value measuring oscillation magnitude of the volage at the agent position.
                last_action_onehot: the last timestep action under one-hot encoding form.

                one-hot encoding: an array of zeros everywhere and have a value of 1 at the executed position.
                For example, we discretize the action space into DISCRETIZE=10 bins, and the action sent back from
                RLlib is: 3. The one-hot encoding of the action is: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    """

    @property
    def observation_space(self):
        obss = self.env.observation_space
        acts = self.env.action_space
        if type(obss) is Box:
            shpobs = obss.shape
            shpact = acts.shape

        observation_space = Dict({
            "own_obs": Box(low=-float('inf'), high=float('inf'), shape=(shpobs[0] + 13,), dtype=np.float64),
            "opponent_obs": Box(low=-float('inf'), high=float('inf'), shape=(12 * (shpobs[0] + 13),), dtype=np.float64),
            "opponent_action": Box(low=-float('inf'), high=float('inf'), shape=(12 * DISCRETIZE_RELATIVE,),
                                   dtype=np.float64),
        })
        return observation_space

    def observation(self, observation, info):
        global_obs = {}
        rl_ids = list(observation.keys())
        rl_ids.sort()

        for rl_id in observation.keys():
            onehot_ids = [0.] * len(rl_ids)
            index_id = rl_ids.index(rl_id)
            onehot_ids[index_id] = 1.
            observation[rl_id] = list(observation[rl_id]) + list(onehot_ids)

        for rl_id in observation.keys():
            other_rl_ids = list(observation.keys())
            other_rl_ids.remove(rl_id)
            own_obs = observation[rl_id]
            opponent_obs = []

            for other_rl_id in other_rl_ids:
                opponent_obs += list(observation[other_rl_id])

            global_obs[rl_id] = {"own_obs": own_obs,
                                 "opponent_obs": np.array(opponent_obs),
                                 "opponent_action": np.array([-1.] * 12 * DISCRETIZE_RELATIVE)  # case of single action
                                 }
        return global_obs
