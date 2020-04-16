import numpy as np

from pycigar.envs.wrappers.wrapper import Wrapper
from pycigar.utils.logging import logger


class RewardWrapper(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_r = self.reward(reward, info)
        Logger = logger()
        reward_dict = new_r if isinstance(new_r, dict) else {k: new_r for k in self.k.device.get_rl_device_ids()}
        for k, r in reward_dict.items():
            Logger.log(k, 'reward', r)

        return observation, new_r, done, info

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
        raise NotImplementedError


class LocalRewardWrapper(RewardWrapper):
    """
    The reward for each agent. It depends on agent local observation.
    """

    def reward(self, reward, info):
        A = 0  # weight for voltage in reward function
        B = 100  # weight for y-value in reward function
        C = 1  # weight for the percentage of power injection
        D = 1  # weight for taking different action from last timestep action
        E = 5  # weight for taking different action from the initial action

        rewards = {}
        # for each agent, we set the reward as under, note that agent reward is only a function of local information.
        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = self.INIT_ACTION[key]
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = self.INIT_ACTION[key]
            voltage = info[key]['voltage']
            y = info[key]['y']
            p_inject = info[key]['p_inject']
            p_max = info[key]['p_max']
            r = -(np.sqrt(A * (1 - voltage) ** 2 + B * y ** 2 + C * (1 + p_inject / p_max) ** 2) + D * np.sum(
                (action - old_action) ** 2))
            rewards.update({key: r})

        return rewards


class GlobalRewardWrapper(RewardWrapper):
    def reward(self, reward, info):
        M = self.sim_params['M']
        N = self.sim_params['N']
        P = self.sim_params['P']

        rewards = {}
        global_reward = 0
        # we accumulate agents reward into global_reward and divide it with the number of agents.
        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = self.INIT_ACTION[key]
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = self.INIT_ACTION[key]
            y = info[key]['y']
            r = 0
            r = -((M * y ** 2 + N * np.sum((action - old_action) ** 2) + P * np.sum(
                (action - self.INIT_ACTION[key]) ** 2))) / 100
            global_reward += r
        global_reward = global_reward / len(list(info.keys()))
        for key in info.keys():
            rewards.update({key: global_reward})

        return rewards


class SecondStageGlobalRewardWrapper(RewardWrapper):
    def reward(self, reward, info):
        rewards = {}
        global_reward = 0
        # we accumulate agents reward into global_reward and divide it with the number of agents.

        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = self.INIT_ACTION[key]
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = self.INIT_ACTION[key]
            y = info[key]['y']

            r = 0

            if (action == old_action).all():
                roa = 0
            else:
                roa = 1
            if (action == self.INIT_ACTION[key]).all():
                ria = 0
            else:
                ria = 1

            r += -(2 * y + 0.1 * roa + np.linalg.norm(action - self.INIT_ACTION[key]))
            # r += -((M2*y**2 + P2*np.sum(np.abs(action-old_action)) + N2*np.sum(np.abs(action-INIT_ACTION))))/100
            global_reward += r
        global_reward = global_reward / len(list(info.keys()))
        for key in info.keys():
            rewards.update({key: global_reward})

        return rewards


class SearchGlobalRewardWrapper(RewardWrapper):
    def reward(self, reward, info):
        M = self.sim_params['M']
        N = self.sim_params['N']
        P = self.sim_params['P']
        rewards = {}
        global_reward = 0
        # we accumulate agents reward into global_reward and divide it with the number of agents.
        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = self.INIT_ACTION[key]
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = self.INIT_ACTION[key]
            y = info[key]['y']

            r = 0
            # if y > 0.025:
            #    r = -500
            r += -((M * y ** 2 + N * np.sum((action - old_action) ** 2) + P * np.sum(
                (action - self.INIT_ACTION[key]) ** 2))) / 100
            global_reward += r
        global_reward = global_reward / len(list(info.keys()))
        for key in info.keys():
            rewards.update({key: global_reward})

        return rewards


class CentralGlobalRewardWrapper(RewardWrapper):
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

    def __init__(self, env, unbalance=False):
        super().__init__(env)
        self.unbalance = unbalance

    def reward(self, reward, info):
        M = self.k.sim_params['M']
        N = self.k.sim_params['N']
        P = self.k.sim_params['P']

        global_reward = 0
        # we accumulate agents reward into global_reward and divide it with the number of agents.
        if self.unbalance:
            y_or_u = max([info[k]['u'] for k in info.keys()])
        else:
            y_or_u = max([info[k]['y'] for k in info.keys()])

        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = self.INIT_ACTION[key]
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = self.INIT_ACTION[key]

            r = 0

            if (action == old_action).all():
                roa = 0
            else:
                roa = 1

            r += -(M * info[key]['y'] + N * roa + P * np.linalg.norm(action - self.INIT_ACTION[key]) + 0.5 * (
                        1 - abs(info[key]['p_set_p_max'])) ** 2)
            global_reward += r
        global_reward = global_reward / len(list(info.keys()))

        return global_reward
