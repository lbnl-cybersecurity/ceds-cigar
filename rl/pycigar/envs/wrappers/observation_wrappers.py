from collections import deque

from gym.spaces import Dict, Box, Tuple, Discrete

from pycigar.envs.wrappers.wrapper import Wrapper
from pycigar.envs.wrappers.wrappers_constants import *


class ObservationWrapper(Wrapper):
    def reset(self):
        observation = self.env.reset()
        self.INIT_ACTION = self.INIT_ACTION
        return self.observation(observation, info=None)

    def step(self, rl_actions):
        observation, reward, done, info = self.env.step(rl_actions)
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


class LocalObservationWrapper(ObservationWrapper):
    """Observation: a dictionary of local observation for each agent, in the form of {'id_1': obs1, 'id_2': obs2,...}.
                    each agent observation is an array of [voltage, solar_generation, y, p_inject, q_inject]
                    at current timestep,
    """

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float64)

    def observation(self, observation, info):
        return observation


class LocalObservationV2Wrapper(ObservationWrapper):
    """ ATTENTION: this wrapper is only used with single head action wrappers (control only 1 breakpoint).
    Observation: a dictionary of local observation for each agent, in the form of {'id_1': obs1, 'id_2': obs2,...}.
                 each agent observation is an array of [y-value, init_action_onehot, last_action_onehot]
                 at current timestep.

                y_value: the value measuring oscillation magnitude of the volage at the agent position.
                init_action_onehot: the initial action under one-hot encoding form.
                last_action_onehot: the last timestep action under one-hot encoding form.

                one-hot encoding: an array of zeros everywhere and have a value of 1 at the executed position.
                For example, we discretize the action space into DISCRETIZE=10 bins, and the action sent back from
                RLlib is: 3. The one-hot encoding of the action is: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    """

    @property
    def observation_space(self):
        """Define the observation space.
        This is required to have an valid openAI gym environment.

        Returns
        -------
        Box
            A valid observation is an array have range from -inf to inf. y-value is scalar, init_action_onehot
            and last_action_onehot have a size of DISCRETIZE, therefore the shape is (1+2*DISCRETIZE, ).
        """
        return Box(low=-float('inf'), high=float('inf'), shape=(1 + 2 * DISCRETIZE,), dtype=np.float64)

    def observation(self, observation, info):
        # get the old action of last timestep of the agents.
        for key in observation.keys():

            # at reset time, old_action is empty, we set the old_action to init_action.
            if info is None or info[key]['old_action'] is None:
                old_action = self.INIT_ACTION[key]
            else:
                old_action = info[key]['old_action']

            a = int((old_action[1] - ACTION_LOWER_BOUND) / (ACTION_UPPER_BOUND - ACTION_LOWER_BOUND) * DISCRETIZE)
            # creating an array of zero everywhere
            old_action = np.zeros(DISCRETIZE)
            # set value 1 at the executed action, at this step, we have the init_action_onehot.
            old_action[a] = 1

            a = int((self.INIT_ACTION[key][1] - ACTION_LOWER_BOUND) / (
                    ACTION_UPPER_BOUND - ACTION_LOWER_BOUND) * DISCRETIZE)
            # creating an array of zero everywhere
            init_action = np.zeros(DISCRETIZE)
            # set value 1 at the executed action, at this step, we have the init_action_onehot.
            init_action[a] = 1

            # in the original observation, position 2 is the y-value. We concatenate it with init_action and old_action
            observation[key] = np.concatenate((np.array([observation[key][2]]), init_action, old_action))

        return observation


class LocalObservationV3Wrapper(ObservationWrapper):
    """ ATTENTION: this wrapper is only used with single head RELATIVE ACTION wrappers (control only 1 breakpoint).
    Observation: a dictionary of local observation for each agent, in the form of {'id_1': obs1, 'id_2': obs2,...}.
                 each agent observation is an array of [y-value, init_action_onehot, last_action_onehot]
                 at current timestep.

                y_value: the value measuring oscillation magnitude of the volage at the agent position.
                init_action_onehot: the initial action under one-hot encoding form.
                last_action_onehot: the last timestep action under one-hot encoding form.

                one-hot encoding: an array of zeros everywhere and have a value of 1 at the executed position.
                For example, we discretize the action space into DISCRETIZE=10 bins, and the action sent back from
                RLlib is: 3. The one-hot encoding of the action is: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    """

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(2 + 2 * DISCRETIZE_RELATIVE,), dtype=np.float64)

    def observation(self, observation, info):
        a = int(ACTION_RANGE / ACTION_STEP)
        # creating an array of zero everywhere
        init_action = np.zeros(DISCRETIZE_RELATIVE)
        # set value 1 at the executed action, at this step, we have the init_action_onehot.
        init_action[a] = 1

        # get the old action of last timestep of the agents.
        for key in observation.keys():

            # at reset time, old_action is empty, we set the old_action to init_action.
            if info is None or info[key]['old_action'] is None:
                old_action = self.INIT_ACTION[key]
            else:
                old_action = info[key]['old_action']

            a = int((old_action[1] - self.INIT_ACTION[key][1] + ACTION_RANGE) / ACTION_STEP)

            # act = INIT_ACTION - ACTION_RANGE + ACTION_STEP*act
            # creating an array of zero everywhere
            old_action = np.zeros(DISCRETIZE_RELATIVE)
            # set value 1 at the executed action, at this step, we have the init_action_onehot.
            old_action[a] = 1

            # in the original observation, position 2 is the y-value. We concatenate it with init_action and old_action
            if info is None or info[list(info.keys())[0]]['env_time'] < 940:
                hack = np.array([0])
            else:
                hack = np.array([1])

            observation[key] = np.concatenate((np.array([observation[key][2]]), hack, init_action, old_action))

        return observation


class LocalObservationV4Wrapper(ObservationWrapper):
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

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(1 + DISCRETIZE_RELATIVE,), dtype=np.float64)

    def observation(self, observation, info):
        a = int(ACTION_RANGE / ACTION_STEP)
        # creating an array of zero everywhere
        init_action = np.zeros(DISCRETIZE_RELATIVE)
        # set value 1 at the executed action, at this step, we have the init_action_onehot.
        init_action[a] = 1

        # get the old action of last timestep of the agents.
        for key in observation.keys():

            # at reset time, old_action is empty, we set the old_action to init_action.
            if info is None or info[key]['old_action'] is None:
                old_action = self.INIT_ACTION[key]
            else:
                old_action = info[key]['old_action']

            a = int((old_action[1] - self.INIT_ACTION[key][1] + ACTION_RANGE) / ACTION_STEP)
            # act = INIT_ACTION - ACTION_RANGE + ACTION_STEP*act
            # creating an array of zero everywhere
            old_action = np.zeros(DISCRETIZE_RELATIVE)
            # set value 1 at the executed action, at this step, we have the init_action_onehot.
            old_action[a] = 1

            # in the original observation, position 2 is the y-value. We concatenate it with init_action and old_action
            observation[key] = np.concatenate((1e6 * np.array([observation[key][2]]), old_action))

        return observation


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


class FramestackObservationWrapper(ObservationWrapper):
    """Observation: a dictionary of stacked local observation for each agent,
                    in the form of {'id_1': np.array([obs1_t0, obs1_t1, obs1_t2]),
                                    'id_2': np.array([obs2_t0, obs2_t1, obs2_t2]),...}.
    The wrapper is used to stack observation within range of num_frames (number of timesteps).

    Attributes
    ----------
    num_frames : int
       The number of frames (number of timesteps) that the agent will see.

    """

    @property
    def observation_space(self):

        # get the observation space of the last wrapper.
        obss = self.env.observation_space
        # get the shape of the observation of the last wrapper.
        shp = obss.shape

        if type(obss) is Box:
            # we declare the number of frames, so we always keeps num_frames lastest observation form environment.
            self.num_frames = NUM_FRAMES
            # frames is a deque has length num_frames, the latest observation will be at position num_frames-1.
            # deque will automatically pop out the first element (oldest observation)
            # when it is full and we append another element.
            self.frames = deque([], maxlen=self.num_frames)

            # the new observation space will have the shape of (obs_length, num_frames)
            obss = Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(shp[0], self.num_frames),
                dtype=np.float64)

        return obss

    def reset(self):
        # get the observation from environment as usual
        observation = self.env.reset()
        # add the observation into frames
        self.frames.append(observation)

        # forward enviroment num_frames-1 time to fill the frames. (rl action is null).
        for _ in range(self.num_frames - 1):
            observation, _, _, _ = self.env.step({})
            self.frames.append(observation)

        # return the post-process observation
        return self._get_ob()

    def observation(self, observation, info):
        # everytime we get eh new observation from the environment, we add it to the frame and return the
        # post-process observation.
        self.frames.append(observation)
        return self._get_ob()

    def _get_ob(self):
        """The post-process function.
           We need to tranform observation collected in the deque into valid observation.
        Returns
        -------
        dict
            A dictionary of observation that we want.
        """

        # sanity check: ensure the frames has num_frames length.
        assert len(self.frames) == self.num_frames

        obss = self.env.observation_space
        shp = obss.shape

        # get the list of agent ids.
        ids = self.frames[0].keys()
        # initialize new observation.
        obs = {}

        # iterate through frame and stack them.
        for fr in self.frames:
            for i in ids:
                # since we initialize the observation as an empty dictionary, we need to check the appearance of
                # ids first time we add the observation.
                if i not in obs.keys():
                    obs[i] = fr[i].reshape(shp[0], 1)
                else:
                    obs[i] = np.concatenate((obs[i], fr[i].reshape(shp[0], 1)), axis=1)
        return obs


class FramestackObservationV2Wrapper(ObservationWrapper):
    @property
    def observation_space(self):
        """Observation space definition for this wrapper.

        Returns
        -------
        Box
            Observation space for this wrapper.
        """
        obss = self.env.observation_space
        if type(obss) is Box:
            self.num_frames = NUM_FRAMES
            self.frames = deque([], maxlen=self.num_frames)
            shp = obss.shape
            obss = Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(shp[0] + 1,),
                dtype=np.float64)
        return obss

    def reset(self):

        # get the observation from environment as usual
        self.num_frames = NUM_FRAMES
        self.frames = deque([], maxlen=self.num_frames)
        observation = self.env.reset()
        # add the observation into frames
        self.frames.append(observation)
        # forward enviroment num_frames time to fill the frames. (rl action is null).
        for _ in range(self.num_frames - 1):
            observation, _, _, _ = self.env.step({})
            self.frames.append(observation)
        return self._get_ob()

    def observation(self, observation, info):
        # everytime we get eh new observation from the environment, we add it to the frame and return the
        # post-process observation.
        self.frames.append(observation)
        return self._get_ob()

    def _get_ob(self):
        """The post-process function.
           We need to tranform observation collected in the deque into valid observation.
        Returns
        -------
        dict
            A dictionary of observation that we want.
        """

        # sanity check: ensure the frames has num_frames length.
        assert len(self.frames) == self.num_frames
        obss = self.env.observation_space
        shp = obss.shape

        # get the list of agent ids.
        ids = self.frames[0].keys()
        # initialize new observation.
        obs = {}

        """for j in range(1, self.frames):
                                    for i in ids:
                                        if i not in obs.keys():
                                            obs[i] = np.concatenate((self.frames[j][i].reshape(shp[0], 1), (self.frames[j][i][0]-self.frames[j-1][i][0]).reshape(1, 1)), axis=0)
                                        else:
                                            obs_temp = np.concatenate((self.frames[j][i].reshape(shp[0], 1), (self.frames[j][i][0]-self.frames[j-1][i][0]).reshape(1, 1)), axis=0)
                                            obs[i] = np.concatenate((obs[i], obs_temp), axis=1)"""

        # the new observation is the observation from LocalObservationV2Wrapper concatenated with y_value - y_value_at_t-5
        for i in ids:
            if i not in obs.keys():
                obs[i] = np.concatenate((self.frames[self.num_frames - 1][i].reshape(shp[0], 1),
                                         (self.frames[self.num_frames - 1][i][0] - self.frames[0][i][0]).reshape(1, 1)),
                                        axis=0).reshape(shp[0] + 1, )
            else:
                obs[i] = np.concatenate((self.frames[self.num_frames - 1][i].reshape(shp[0], 1),
                                         (self.frames[self.num_frames - 1][i][0] - self.frames[0][i][0]).reshape(1, 1)),
                                        axis=0).reshape(shp[0] + 1, )

        return obs


class FramestackObservationV3Wrapper(ObservationWrapper):
    """ATTENTION: this wrapper is only used after the LocalObservationV2Wrapper/LocalObservationV3Wrapper.
    The return value of this wrapper is:
        [y_value_max, y_value, last_action_onehot, y_value - y_value_(t-5)]

    Attributes
    ----------
    frames : deque
        A deque to keep the lastest 5 observations.
    num_frames : int
        Number of frame we need to keep to have y_value - y_value_(t-5).

    """

    @property
    def observation_space(self):
        obss = self.env.observation_space
        if type(obss) is Box:
            self.num_frames = NUM_FRAMES
            self.frames = deque([], maxlen=self.num_frames)
            self.y_value_max = {}
            shp = obss.shape
            obss = Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(shp[0] + 2,),
                dtype=np.float64)
        return obss

    def reset(self):

        # get the observation from environment as usual
        observation = self.env.reset()
        # add the observation into frames
        self.frames.append(observation)
        # forward enviroment num_frames time to fill the frames. (rl action is null).
        for _ in range(self.num_frames - 1):
            observation, _, _, _ = self.env.step({})
            self.frames.append(observation)
        return self._get_ob()

    def observation(self, observation, info):
        # everytime we get eh new observation from the environment, we add it to the frame and return the
        # post-process observation.
        self.frames.append(observation)
        return self._get_ob()

    def _get_ob(self):
        """The post-process function.
           We need to tranform observation collected in the deque into valid observation.
        Returns
        -------
        dict
            A dictionary of observation that we want.
        """

        # sanity check: ensure the frames has num_frames length.
        assert len(self.frames) == self.num_frames
        obss = self.env.observation_space
        shp = obss.shape

        # get the list of agent ids.
        ids = self.frames[0].keys()

        # get the y_value_max for each observation
        for i in ids:
            y_max_frame = max([frame[i][0] for frame in self.frames])
            if i not in self.y_value_max.keys():
                self.y_value_max[i] = y_max_frame
            else:
                self.y_value_max[i] = max(self.y_value_max[i], y_max_frame)

        # initialize new observation.
        obs = {}

        """for j in range(1, self.frames):
                                    for i in ids:
                                        if i not in obs.keys():
                                            obs[i] = np.concatenate((self.frames[j][i].reshape(shp[0], 1), (self.frames[j][i][0]-self.frames[j-1][i][0]).reshape(1, 1)), axis=0)
                                        else:
                                            obs_temp = np.concatenate((self.frames[j][i].reshape(shp[0], 1), (self.frames[j][i][0]-self.frames[j-1][i][0]).reshape(1, 1)), axis=0)
                                            obs[i] = np.concatenate((obs[i], obs_temp), axis=1)"""

        # the new observation is the observation from LocalObservationV4Wrapper concatenated with y_value - y_value_at_t-5
        for i in ids:
            if i not in obs.keys():
                obs[i] = np.concatenate((self.y_value_max[i].reshape(1, 1),
                                         self.frames[self.num_frames - 1][i].reshape(shp[0], 1),
                                         (self.frames[self.num_frames - 1][i][0] - self.frames[0][i][0]).reshape(1, 1)),
                                        axis=0).reshape(shp[0] + 2, )
            else:
                obs[i] = np.concatenate((self.y_value_max[i].reshape(1, 1),
                                         self.frames[self.num_frames - 1][i].reshape(shp[0], 1),
                                         (self.frames[self.num_frames - 1][i][0] - self.frames[0][i][0]).reshape(1, 1)),
                                        axis=0).reshape(shp[0] + 2, )

        return obs


class FramestackObservationV4Wrapper(ObservationWrapper):
    """ATTENTION: this wrapper is only used after the LocalObservationV2Wrapper/LocalObservationV3Wrapper.
    The return value of this wrapper is:
        [y_value_max, y_value, last_action_onehot, y_value - y_value_(t-5)]

    Attributes
    ----------
    frames : deque
        A deque to keep the lastest 5 observations.
    num_frames : int
        Number of frame we need to keep to have y_value - y_value_(t-5).

    """

    @property
    def observation_space(self):
        obss = self.env.observation_space
        if type(obss) is Box:
            self.num_frames = NUM_FRAMES
            self.frames = deque([], maxlen=self.num_frames)
            self.y_value_max = {}
            shp = obss.shape
            obss = Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(shp[0] + 1,),
                dtype=np.float64)
        return obss

    def reset(self):
        # get the observation from environment as usual
        observation = self.env.reset()
        # add the observation into frames
        self.frames.append(observation)
        # forward enviroment num_frames time to fill the frames. (rl action is null).
        return self._get_ob()

    def observation(self, observation, info):
        # everytime we get eh new observation from the environment, we add it to the frame and return the
        # post-process observation.
        self.frames.append(observation)
        return self._get_ob()

    def _get_ob(self):
        """The post-process function.
           We need to tranform observation collected in the deque into valid observation.
        Returns
        -------
        dict
            A dictionary of observation that we want.
        """
        obss = self.env.observation_space
        shp = obss.shape

        # get the list of agent ids.
        ids = self.frames[0].keys()

        # get the y_value_max for each observation
        for i in ids:
            y_max_frame = max([frame[i][0] for frame in self.frames])
            # if i not in self.y_value_max.keys():
            self.y_value_max[i] = y_max_frame
            # else:
            #    self.y_value_max[i] = max(self.y_value_max[i], y_max_frame)

        # np.array(y_value_max.values())
        # initialize new observation.
        obs = {}

        """for j in range(1, self.frames):
                                    for i in ids:
                                        if i not in obs.keys():
                                            obs[i] = np.concatenate((self.frames[j][i].reshape(shp[0], 1), (self.frames[j][i][0]-self.frames[j-1][i][0]).reshape(1, 1)), axis=0)
                                        else:
                                            obs_temp = np.concatenate((self.frames[j][i].reshape(shp[0], 1), (self.frames[j][i][0]-self.frames[j-1][i][0]).reshape(1, 1)), axis=0)
                                            obs[i] = np.concatenate((obs[i], obs_temp), axis=1)"""

        # the new observation is the observation from LocalObservationV4Wrapper concatenated with y_value - y_value_at_t-5
        # for i in ids:
        #    if i not in obs.keys():
        #        obs[i] = np.concatenate((self.y_value_max[i].reshape(1, 1), self.frames[-1][i].reshape(shp[0], 1)), axis=0).reshape(shp[0]+1, )
        #    else:
        #        obs[i] = np.concatenate((self.y_value_max[i].reshape(1, 1), self.frames[-1][i].reshape(shp[0], 1),), axis=0).reshape(shp[0]+1, )
        for i in ids:
            if i not in obs.keys():
                obs[i] = np.concatenate((self.y_value_max[i].reshape(1, 1), self.frames[-1][i].reshape(shp[0], 1)),
                                        axis=0).reshape(shp[0] + 1, )
            else:
                obs[i] = np.concatenate((self.y_value_max[i].reshape(1, 1), self.frames[-1][i].reshape(shp[0], 1),),
                                        axis=0).reshape(shp[0] + 1, )

        return obs


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
