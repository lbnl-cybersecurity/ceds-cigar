from pycigar.envs.multiagent.wrapper_multi import ObservationWrapper, RewardWrapper, ActionWrapper
from gym.spaces import Dict, Tuple, Discrete, Box
import numpy as np
from collections import deque

DISCRETIZE = 30
INIT_ACTION = np.array([0.98, 1.01, 1.01, 1.04, 1.08])
A = 0
B = 100
C = 1
D = 1
E = 5
###########################################################################
#                         OBSERVATION WRAPPER                             #
###########################################################################


class LocalObservationWrapper(ObservationWrapper):

    """A dictionary of observation of each device.
    """

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32)

    def observation(self, observation):
        # init_action = INIT_ACTION
        # for key in observation.keys():
        #    observation[key] = np.concatenate((observation[key], init_action))
        return observation


class LocalObservationV2Wrapper(ObservationWrapper):

    """Observation is y-value, and one-hot encoding init action, old action.
    """

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(1 + 2*DISCRETIZE, ), dtype=np.float32)

    def observation(self, observation):
        init_action = INIT_ACTION
        a = int((INIT_ACTION[1]-0.8)/(1.1-0.8)*DISCRETIZE)
        init_action = np.zeros(DISCRETIZE)
        init_action[a] = 1  # one-hot encoding for initial action
        for key in observation.keys():
            try:
                old_action = self.get_old_actions()[key]
            except:
                old_action = INIT_ACTION

            a = int((old_action[1]-0.8)/(1.1-0.8)*DISCRETIZE)
            old_action = np.zeros(DISCRETIZE)
            old_action[a] = 1  # one-hot encoding for old action
            observation[key] = np.concatenate((np.array([observation[key][2]]), init_action, old_action))
        return observation


class GlobalObservationWrapper(ObservationWrapper):

    """A dictionary of observation for each device, containing its own observation, opponent observations and their actions.
    """

    @property
    def observation_space(self):
        observation_space = Dict({
                        "own_obs": Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32),
                        "opponent_obs": Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32),
                        "opponent_action": Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32),
                        })
        return observation_space

    def observation(self, observation):
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

    """The wrapper to stack observation within range of num_frames.

    Attributes
    ----------
    num_frames : int
       The number of frames that the agent will see.
    """

    @property
    def observation_space(self):
        obss = self.env.observation_space
        if type(obss) is Box:
            self.num_frames = 4
            self.frames = deque([], maxlen=self.num_frames)
            shp = obss.shape
            obss = Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(shp[0], self.num_frames),
                dtype=np.float32)
        return obss

    def reset(self):
        observation = self.env.reset()
        self.frames.append(observation)
        for _ in range(self.num_frames-1):
            observation, _, _, _ = self.env.step({})
            self.frames.append(observation)
        return self._get_ob()

    def observation(self, observation):
        self.frames.append(observation)
        return self._get_ob()

    def _get_ob(self):
        assert len(self.frames) == self.num_frames
        obss = self.env.observation_space
        shp = obss.shape
        ids = self.frames[0].keys()
        obs = {}
        for fr in self.frames:
            for i in ids:
                if i not in obs.keys():
                    obs[i] = fr[i].reshape(shp[0], 1)
                else:
                    obs[i] = np.concatenate((obs[i], fr[i].reshape(shp[0], 1)), axis=1)
        return obs


class FramestackObservationV2Wrapper(ObservationWrapper):
    @property
    def observation_space(self):
        obss = self.env.observation_space
        if type(obss) is Box:
            self.num_frames = 5
            self.frames = deque([], maxlen=self.num_frames)
            shp = obss.shape
            obss = Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(shp[0]+1, ),
                dtype=np.float32)
        return obss

    def reset(self):
        observation = self.env.reset()
        self.frames.append(observation)
        for _ in range(self.num_frames):
            observation, _, _, _ = self.env.step({})
            self.frames.append(observation)
        return self._get_ob()

    def observation(self, observation):
        self.frames.append(observation)
        return self._get_ob()

    def _get_ob(self):
        assert len(self.frames) == self.num_frames
        obss = self.env.observation_space
        shp = obss.shape
        ids = self.frames[0].keys()
        obs = {}
        """for j in range(1, self.frames):
                                    for i in ids:
                                        if i not in obs.keys():
                                            obs[i] = np.concatenate((self.frames[j][i].reshape(shp[0], 1), (self.frames[j][i][0]-self.frames[j-1][i][0]).reshape(1, 1)), axis=0)
                                        else:
                                            obs_temp = np.concatenate((self.frames[j][i].reshape(shp[0], 1), (self.frames[j][i][0]-self.frames[j-1][i][0]).reshape(1, 1)), axis=0)
                                            obs[i] = np.concatenate((obs[i], obs_temp), axis=1)"""

        for i in ids:
            if i not in obs.keys():
                obs[i] = np.concatenate((self.frames[self.num_frames-1][i].reshape(shp[0], 1), (self.frames[self.num_frames-1][i][0]-self.frames[0][i][0]).reshape(1, 1)), axis=0).reshape(shp[0]+1, )
            else:
                obs[i] = np.concatenate((self.frames[self.num_frames-1][i].reshape(shp[0], 1), (self.frames[self.num_frames-1][i][0]-self.frames[0][i][0]).reshape(1, 1)), axis=0).reshape(shp[0]+1, )

        return obs
###########################################################################
#                            REWARD WRAPPER                               #
###########################################################################


class LocalRewardWrapper(RewardWrapper):

    """The reward of each device, base on the local information.
    """

    def reward(self, reward):
        new_actions = {}
        for rl_id in self.env.k.device.get_rl_device_ids():
            new_actions[rl_id] = self.env.k.device.get_control_setting(rl_id)
        if new_actions == {}:
            return {}

        rewards = {}
        for rl_id in self.env.k.device.get_rl_device_ids():
            action = new_actions[rl_id]
            old_action = self.get_old_actions()[rl_id]
            connected_node = self.env.k.device.get_node_connected_to(rl_id)
            voltage = self.env.k.node.get_node_voltage(connected_node)
            y = self.env.k.device.get_device_y(rl_id)
            p_inject = self.env.k.device.get_device_p_injection(rl_id)
            p_max = self.env.k.device.get_solar_generation(rl_id)
            r = -(np.sqrt(A*(1-voltage)**2 + B*y**2 + C*(1+p_inject/p_max)**2) + D*np.sum((action-old_action)**2))
            rewards.update({rl_id: r})

        return rewards


class GlobalRewardWrapper(RewardWrapper):

    """The reward of each device is the average of all rewards.
    """
    def reward(self, reward):
        new_actions = {}
        for rl_id in self.env.k.device.get_rl_device_ids():
            new_actions[rl_id] = self.env.k.device.get_control_setting(rl_id)
        if new_actions == {}:
            return {}

        rewards = {}

        global_reward = 0
        for rl_id in self.env.k.device.get_rl_device_ids():
            action = np.array(new_actions[rl_id])
            try:
                old_action = np.array(self.env.old_actions[rl_id])
            except:
                old_action = INIT_ACTION

            init_action = INIT_ACTION
            connected_node = self.env.k.device.get_node_connected_to(rl_id)
            voltage = self.env.k.node.get_node_voltage(connected_node)
            y = self.env.k.device.get_device_y(rl_id)
            p_inject = self.env.k.device.get_device_p_injection(rl_id)
            p_max = self.env.k.device.get_solar_generation(rl_id)
            B = 0
            E = 5
            D = 0
            r = -((B*y**2 + D*np.sum((action-old_action)**2) + E*np.sum((action-init_action)**2)))/100
            global_reward += r
        global_reward = global_reward / len(self.env.k.device.get_rl_device_ids())
        for rl_id in self.env.k.device.get_rl_device_ids():
            rewards.update({rl_id: global_reward})

        return rewards


class SecondStageGlobalRewardWrapper(RewardWrapper):

    """The reward of each device is the average of all rewards.
    This reward is more complicated than the GlobalRewardWrapper, the purpose is to do curriculum learning.
    """

    def reward(self, reward):
        new_actions = {}
        for rl_id in self.env.k.device.get_rl_device_ids():
            new_actions[rl_id] = self.env.k.device.get_control_setting(rl_id)
        if new_actions == {}:
            return {}

        rewards = {}

        global_reward = 0
        for rl_id in self.env.k.device.get_rl_device_ids():
            action = np.array(new_actions[rl_id])
            try:
                old_action = np.array(self.env.old_actions[rl_id])
            except:
                old_action = INIT_ACTION

            init_action = INIT_ACTION
            connected_node = self.env.k.device.get_node_connected_to(rl_id)
            voltage = self.env.k.node.get_node_voltage(connected_node)
            y = self.env.k.device.get_device_y(rl_id)
            p_inject = self.env.k.device.get_device_p_injection(rl_id)
            p_max = self.env.k.device.get_solar_generation(rl_id)
            B = 500
            E = 5
            D = 1
            r = -((B*y**2 + D*np.sum((action-old_action)**2) + E*np.sum((action-init_action)**2)))/100
            global_reward += r
        global_reward = global_reward / len(self.env.k.device.get_rl_device_ids())
        for rl_id in self.env.k.device.get_rl_device_ids():
            rewards.update({rl_id: global_reward})

        return rewards


###########################################################################
#                            ACTION WRAPPER                               #
###########################################################################

class SingleDiscreteActionWrapper(ActionWrapper):

    """The action head is 1 action discretized into DISCRETIZE number of bins.
    We control 5 VBPs by translate the VBPs.
    """

    @property
    def action_space(self):
        return Discrete(DISCRETIZE)

    def action(self, action):
        new_action = {}
        for rl_id, act in action.items():
            act = 0.8 + (1.1-0.8)/DISCRETIZE*act
            act = np.array([act-0.03, act, act, act+0.03, act+0.07])
            new_action[rl_id] = act
        return new_action


class SingleRelativeInitDiscreteActionWrapper(ActionWrapper):

    """The action head is 1 action discretized into 10 bins.
    Each bin is a step of 0.02 deviated from the initial action.
    """
    @property
    def action_space(self):
        return Discrete(DISCRETIZE)

    def action(self, action):
        new_action = {}
        init_action = INIT_ACTION
        for rl_id, act in action.items():
            act = init_action - 0.1 + 0.2/DISCRETIZE*act
            new_action[rl_id] = act
        return new_action


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


# AR: AutoRegressive
class ARDiscreteActionWrapper(ActionWrapper):

    """Action head for autoregressive head case.
    """

    @property
    def action_space(self):
        return Tuple([Discrete(DISCRETIZE), Discrete(DISCRETIZE),
                      Discrete(DISCRETIZE), Discrete(DISCRETIZE),
                      Discrete(DISCRETIZE)])

    def action(self, action):
        new_action = {}
        for rl_id, act in action.items():
            act = 0.8 + (1.1-0.8)/DISCRETIZE*np.array(act, np.float32)
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
