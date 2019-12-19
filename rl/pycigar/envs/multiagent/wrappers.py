from pycigar.envs.multiagent.wrapper_multi import ObservationWrapper, RewardWrapper, ActionWrapper
from gym.spaces import Dict, Tuple, Discrete, Box
import numpy as np
from collections import deque

# action space discretization
DISCRETIZE = 30
# the initial action for inverter
INIT_ACTION = np.array([0.98, 1.01, 1.01, 1.04, 1.08])

A = 0       # weight for voltage in reward function
B = 100     # weight for y-value in reward function
C = 1       # weight for the percentage of power injection
D = 1       # weight for taking different action from last timestep action
E = 5       # weight for taking different action from the initial action

# params for local reward wrapper, simple reward
M = 0  # weight for y-value in reward function
N = 10  # weight for taking different action from the initial action
P = 0  # weight for taking different action from last timestep action

# params for local reward wrapper, complex reward
M2 = 10  # weight for y-value in reward function
N2 = 10  # weight for taking different action from the initial action
P2 = 2  # weight for taking different action from last timestep action


# single head action
ACTION_CURVE = np.array([-0.03, 0., 0., 0.03, 0.07])
ACTION_LOWER_BOUND = 0.8
ACTION_UPPER_BOUND = 1.1

# relative single head action
ACTION_RANGE = 0.1
ACTION_STEP = 0.02
DISCRETIZE_RELATIVE = int((ACTION_RANGE/ACTION_STEP))*2 + 1

# number of frames to keep
NUM_FRAMES = 5

"""
In the original multiagent environment (the detail implementation is in pycigar/envs/multiagent/base.py and pycigar/envs/base.py),
the details for observation , action and reward are as following:

    - Observation: a dictionary of local observation for each agent, in the form of {'id_1': obs1, 'id_2': obs2,...}.
                   each agent observation is an array of [voltage, solar_generation, y, p_inject, q_inject]
                   at current timestep,
      Example:
        >>> {'pv_5': array([ 1.02470216e+00,  6.73415767e+01,  2.37637525e-02, -6.70097474e+01, 2.28703002e+01]),
             'pv_6': array([ 1.02461386e+00,  7.42201160e+01,  2.36835291e-02, -7.42163505e+01, 2.35234973e+01]), ...}

      To have a valid openAI gym environment, we have to declare the observation space, what value and dimension will be valid as
      a right observation.

      The observation space is:
        Box(low=-float('inf'), high=float('inf'), shape=(5, ), dtype=np.float32),
      which describe an array of 5, each value can range from -infinity to infinity.

    - Reward: a dictionary of reward for each agent, in the form of {'id_1': reward1, 'id_2': reward2,...}
              default reward for each agent is 0. Depending on our need, we will write suitable wrappers.
      Example:
        >>> {'pv_5': 0, 'pv_6': 0,...}

    - Action: a dictionary of action for each agent, in for form of {'id_1': act1, 'id_2': act2,...}
              there are 2 forms of action: the actions return from RLlib and the actions we use to feed into our environment.

              the actions return from RLlib can be anything (only one value of controlled breakpoint,
              or 5-discretized values of controlled breakpoints) but before we feed the actions to the environment,
              we transform it into the valid form which the environment can execute. For inverter, the control is
              an array of 5 breakpoints which is the actual setting of inverter that we want.
              A valid form of action to feed to the environment: {'id_1': arrayof5, 'id_2': arrayof5,...}
      Example:
        >>> {'pv_5': anp.array([0.98, 1.01, 1.01, 1.04, 1.08]),
             'pv_6': np.array([0.90, 1.00, 1.02, 1.03, 1.07]), ...}

Base on the original multiagent environment observation, action and reward,
the wrappers are used to change the observation, action and reward as we need for different experiment.
"""

###########################################################################
#                         OBSERVATION WRAPPER                             #
###########################################################################
"""
Wrappers for custom observation.
"""


class LocalObservationWrapper(ObservationWrapper):

    """Observation: a dictionary of local observation for each agent, in the form of {'id_1': obs1, 'id_2': obs2,...}.
                    each agent observation is an array of [voltage, solar_generation, y, p_inject, q_inject]
                    at current timestep,
    """

    @property
    def observation_space(self):
        """Define the observation space.
        This is required to have an valid openAI gym environment.

        Returns
        -------
        Box
            A valid observation is an array of 5 values which have range from -inf to inf.
        """
        return Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32)

    def observation(self, observation, info):
        """Modifying the original observation into the observation that we want.

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
        return Box(low=-float('inf'), high=float('inf'), shape=(1 + 2*DISCRETIZE, ), dtype=np.float32)

    def observation(self, observation, info):
        """Modifying the original observation into the observation that we want.

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

        # tranform back the initial action to the action form of RLlib

        a = int((INIT_ACTION[1]-ACTION_LOWER_BOUND)/(ACTION_UPPER_BOUND-ACTION_LOWER_BOUND)*DISCRETIZE)
        # creating an array of zero everywhere
        init_action = np.zeros(DISCRETIZE)
        # set value 1 at the executed action, at this step, we have the init_action_onehot.
        init_action[a] = 1

        # get the old action of last timestep of the agents.
        for key in observation.keys():

            # at reset time, old_action is empty, we set the old_action to init_action.
            if info is None or info[key]['old_action'] is None:
                old_action = INIT_ACTION
            else:
                old_action = info[key]['old_action']

            # tranform back the initial action to the action form of RLlib
            a = int((old_action[1]-ACTION_LOWER_BOUND)/(ACTION_UPPER_BOUND-ACTION_LOWER_BOUND)*DISCRETIZE)
            # creating an array of zero everywhere
            old_action = np.zeros(DISCRETIZE)
            # set value 1 at the executed action, at this step, we have the init_action_onehot.
            old_action[a] = 1

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
        """Define the observation space.
        This is required to have an valid openAI gym environment.

        Returns
        -------
        Box
            A valid observation is an array have range from -inf to inf. y-value is scalar, init_action_onehot
            and last_action_onehot have a size of DISCRETIZE_RELATIVE, therefore the shape is (1+2*DISCRETIZE_RELATIVE, ).
        """
        return Box(low=-float('inf'), high=float('inf'), shape=(2 + 2*DISCRETIZE_RELATIVE, ), dtype=np.float32)

    def observation(self, observation, info):
        """Modifying the original observation into the observation that we want.

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
        # tranform back the initial action to the action form of RLlib
        a = int(ACTION_RANGE/ACTION_STEP)
        # creating an array of zero everywhere
        init_action = np.zeros(DISCRETIZE_RELATIVE)
        # set value 1 at the executed action, at this step, we have the init_action_onehot.
        init_action[a] = 1

        # get the old action of last timestep of the agents.
        for key in observation.keys():

            # at reset time, old_action is empty, we set the old_action to init_action.
            if info is None or info[key]['old_action'] is None:
                old_action = INIT_ACTION
            else:
                old_action = info[key]['old_action']

            # tranform back the initial action to the action form of RLlib
            a = int((old_action[1]-INIT_ACTION[1]+ACTION_RANGE)/ACTION_STEP)

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


class GlobalObservationWrapper(ObservationWrapper):

    @property
    def observation_space(self):
        observation_space = Dict({
                        "own_obs": Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32),
                        "opponent_obs": Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32),
                        "opponent_action": Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32),
                        })
        return observation_space

    def observation(self, observation, info):
        global_obs = {}
        for rl_id in observation.keys():
            other_rl_ids = list(observation.keys())
            other_rl_ids.remove(rl_id)
            own_obs = observation[rl_id]
            opponent_obs = []

            for other_rl_id in other_rl_ids:
                opponent_obs += list(observation[other_rl_id])

            global_obs[rl_id] = {"own_obs": own_obs,
                                 "opponent_obs": np.array(opponent_obs),
                                 "opponent_action": np.array([-1., -1., -1., -1., -1.])
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
        """Observation space for framestack observation wrapper.
        The shape of observation for each agent is (obs_length, num_frames).

        Returns
        -------
        dict
            A dictionary of framestack observation.
        """

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
                dtype=np.float32)

        return obss

    def reset(self):
        """Redefine reset of the environment.
        This is a must since we need to warm up the deque frames, it needs to have enough num_frames value.

        Returns
        -------
        dict
            A dictionary of post-process observation after reset.
        """

        # get the observation from environment as usual
        observation = self.env.reset()
        # add the observation into frames
        self.frames.append(observation)

        # forward enviroment num_frames-1 time to fill the frames. (rl action is null).
        for _ in range(self.num_frames-1):
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

    """ATTENTION: this wrapper is only used after the LocalObservationV2Wrapper/LocalObservationV3Wrapper.
    The return value of this wrapper is:
        [y_value, init_action_onehot, last_action_onehot, y_value - y_value_(t-5)]

    Attributes
    ----------
    frames : deque
        A deque to keep the lastest 5 observations.
    num_frames : int
        Number of frame we need to keep to have y_value - y_value_(t-5).

    """

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
                shape=(shp[0]+1, ),
                dtype=np.float32)
        return obss

    def reset(self):
        """Redefine reset of the environment.
        This is a must since we need to warm up the deque frames, it needs to have enough num_frames value.

        Returns
        -------
        dict
            A dictionary of post-process observation after reset.
        """

        # get the observation from environment as usual
        self.num_frames = NUM_FRAMES
        self.frames = deque([], maxlen=self.num_frames)
        observation = self.env.reset()
        # add the observation into frames
        self.frames.append(observation)
        # forward enviroment num_frames time to fill the frames. (rl action is null).
        for _ in range(self.num_frames-1):
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
                obs[i] = np.concatenate((self.frames[self.num_frames-1][i].reshape(shp[0], 1), (self.frames[self.num_frames-1][i][0]-self.frames[0][i][0]).reshape(1, 1)), axis=0).reshape(shp[0]+1, )
            else:
                obs[i] = np.concatenate((self.frames[self.num_frames-1][i].reshape(shp[0], 1), (self.frames[self.num_frames-1][i][0]-self.frames[0][i][0]).reshape(1, 1)), axis=0).reshape(shp[0]+1, )

        return obs

###########################################################################
#                            REWARD WRAPPER                               #
###########################################################################


"""
Wrappers for custom reward.
"""


class LocalRewardWrapper(RewardWrapper):

    """The reward for each agent. It depdends on agent local observation.
    """

    def reward(self, reward, info):
        """Redefine the reward of the last wrapper.
        Local reward.
        Parameters
        ----------
        reward : dict
            Dictionary of reward from the last wrapper.
        info : dict
            additional information returned by environment.

        Returns
        -------
        dict
            A dictionary of new rewards.
        """

        rewards = {}
        # for each agent, we set the reward as under, note that agent reward is only a function of local information.
        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = INIT_ACTION
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = INIT_ACTION
            voltage = info[key]['voltage']
            y = info[key]['y']
            p_inject = info[key]['p_inject']
            p_max = info[key]['p_max']
            r = -(np.sqrt(A*(1-voltage)**2 + B*y**2 + C*(1+p_inject/p_max)**2) + D*np.sum((action-old_action)**2))
            rewards.update({key: r})

        return rewards


class GlobalRewardWrapper(RewardWrapper):

    """Redefine the reward of the last wrapper.
    Global reward: reward of each agent is the average of reward from all agents.

    For instance, reward is to encourage the agent not to take action different from the initial action.
    Parameters
    ----------
    reward : dict
        Dictionary of reward from the last wrapper.
    info : dict
        additional information returned by environment.

    Returns
    -------
    dict
        A dictionary of new rewards.
    """

    def reward(self, reward, info):

        rewards = {}

        global_reward = 0

        # we accumulate agents reward into global_reward and devide it with the number of agents.
        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = INIT_ACTION
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = INIT_ACTION
            y = info[key]['y']
            r = 0
            r = -((M*y**2 + P*np.sum((action-old_action)**2) + N*np.sum((action-INIT_ACTION)**2)))/100
            global_reward += r
        global_reward = global_reward / len(list(info.keys()))
        for key in info.keys():
            rewards.update({key: global_reward})

        return rewards


class SecondStageGlobalRewardWrapper(RewardWrapper):

    """Redefine the reward of the last wrapper.
    Global reward: reward of each agent is the average of reward from all agents.

    For instance, reward is to encourage the agent not to take action when unnecessary and damping the oscillation.
    Parameters
    ----------
    reward : dict
        Dictionary of reward from the last wrapper.
    info : dict
        additional information returned by environment.

    Returns
    -------
    dict
        A dictionary of new rewards.
    """

    def reward(self, reward, info):
        rewards = {}
        global_reward = 0
        # we accumulate agents reward into global_reward and devide it with the number of agents.
        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = INIT_ACTION
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = INIT_ACTION
            y = info[key]['y']
            r = 0
            r += -((M2*y**2 + P2*np.sum(np.abs(action-old_action)) + N2*np.sum(np.abs(action-INIT_ACTION))))/100
            global_reward += r
        global_reward = global_reward / len(list(info.keys()))
        for key in info.keys():
            rewards.update({key: global_reward})

        return rewards

class SearchGlobalRewardWrapper(RewardWrapper):

    """Redefine the reward of the last wrapper.
    Global reward: reward of each agent is the average of reward from all agents.

    For instance, reward is to encourage the agent not to take action when unnecessary and damping the oscillation.
    Parameters
    ----------
    reward : dict
        Dictionary of reward from the last wrapper.
    info : dict
        additional information returned by environment.

    Returns
    -------
    dict
        A dictionary of new rewards.
    """

    def reward(self, reward, info):
        rewards = {}
        global_reward = 0
        # we accumulate agents reward into global_reward and devide it with the number of agents.
        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = INIT_ACTION
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = INIT_ACTION
            y = info[key]['y']

            r = 0
            #if y > 0.025:
            #    r = -500
            r += -((self.env.sim_params['M2']*y**2 + self.env.sim_params['P2']*np.sum((action-old_action)**2) + self.env.sim_params['N2']*np.sum((action-INIT_ACTION)**2)))/100
            global_reward += r
        global_reward = global_reward / len(list(info.keys()))
        for key in info.keys():
            rewards.update({key: global_reward})

        return rewards

###########################################################################
#                            ACTION WRAPPER                               #
###########################################################################

"""
Wrappers for custom action head.
"""


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

    def action(self, action):
        """Modify action before feed into the simulation.

        Parameters
        ----------
        action : dict
            The action value for each agent we received from RLlib. Each action is an interger range(0, DISCRETIZE)

        Returns
        -------
        dict
            Action value for each agent with a valid form to feed into the environment.
        """
        new_action = {}
        for rl_id, act in action.items():
            act = ACTION_LOWER_BOUND + (ACTION_UPPER_BOUND-ACTION_LOWER_BOUND)/DISCRETIZE*act
            act = ACTION_CURVE + act
            new_action[rl_id] = act
        return new_action


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

    def action(self, action):
        """Modify action before feed into the simulation.

        Parameters
        ----------
        action : dict
            The action value for each agent we received from RLlib. Each action is an interger range(0, DISCRETIZE)

        Returns
        -------
        dict
            Action value for each agent with a valid form to feed into the environment.
        """
        new_action = {}
        for rl_id, act in action.items():
            act = INIT_ACTION - ACTION_RANGE + ACTION_STEP*act
            new_action[rl_id] = act
        return new_action


# AR: AutoRegressive
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

    def action(self, action):
        """Modify action before feed into the simulation.

        Parameters
        ----------
        action : dict
            The action value for each agent we received from RLlib. Each action is an array of 5, value range(0, DISCRETIZE)

        Returns
        -------
        dict
            Action value for each agent with a valid form to feed into the environment.
        """
        new_action = {}
        for rl_id, act in action.items():
            # This is used to form the discretized value into the valid action before feed into the environment.
            act = ACTION_LOWER_BOUND + (ACTION_UPPER_BOUND-ACTION_LOWER_BOUND)/DISCRETIZE*np.array(act, np.float32)
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
            new_action[rl_id] = act
        return new_action


class ARContinuousActionWrapper(ActionWrapper):
    pass


class SingleContinuousActionWrapper(ActionWrapper):

    """Have not been used.
    """

    @property
    def action_space(self):
        return Box(low=0.8, high=1.1, shape=(1,), dtype=np.float32)

    def action(self, action):
        new_action = {}
        for rl_id, act in action.items():
            act = np.array([act-0.01, act, act, act+0.01, act+0.02])
            new_action[rl_id] = act
        return new_action


