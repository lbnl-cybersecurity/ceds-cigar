from collections import deque

from gym.spaces import Box, Tuple, Discrete
from pycigar.envs.wrappers.wrapper import Wrapper
from pycigar.envs.wrappers.wrappers_constants import *
from pycigar.utils.logging import logger
from pycigar.envs.wrappers.observation_wrappers import ObservationWrapper
from pycigar.envs.exp_envs.env import DELAY

NUM_FRAMES = 150

class DelayObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
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

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(6 + self.a_size, ), dtype=np.float64)

    def observation(self, observation, info):
        if info:
            old_actions = info[list(info.keys())[0]]['raw_action']
            p_set = np.mean([1.5e-3 * info[k]['sbar_solar_irr'] for k in self.k.device.get_rl_device_ids()])
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

        va = (observation['v_worst'][0]-1)*10*2
        vb = (observation['v_worst'][1]-1)*10*2
        vc = (observation['v_worst'][2]-1)*10*2
        observation = np.array([observation['y']*10, observation['u_worst']*10, p_set, *old_a_encoded, va, vb, vc])

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
            obss = Box(low=-float('inf'), high=float('inf'), shape=(shp[0] + 1,), dtype=np.float64)
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

        Logger = logger()
        for _ in range(self.env.k.sim_params['env_config']['sims_per_step']):
            Logger.log('component_observation', 'component_ymax', y_value_max)
        return new_obs