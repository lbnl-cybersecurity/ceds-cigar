import numpy as np

from pycigar.envs.wrappers.wrapper import Wrapper
from pycigar.utils.logging import logger
from gym.spaces import Box

VOLTAGE_THRESHOLD_LB = 0.95

class RewardWrapper(Wrapper):
    def step(self, action, randomize_rl_update=None):
        observation, reward, done, info = self.env.step(action, randomize_rl_update)
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

    def __init__(self, env, unbalance=False):
        super().__init__(env)
        self.unbalance = unbalance

    def reward(self, reward, info):
        M = self.k.sim_params['M']
        N = self.k.sim_params['N']
        P = self.k.sim_params['P']

        rewards = {}
        y_or_u = 'u' if self.unbalance else 'y'
        # for each agent, we set the reward as under, note that agent reward is only a function of local information.

        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = self.INIT_ACTION[key]

            old_action = info[key]['old_action']
            if old_action is None:
                old_action = self.INIT_ACTION[key]

            action = np.array(action)
            old_action = np.array(old_action)
            if (action == old_action).all():
                roa = 0
            else:
                roa = 1

            r = -(
                M * info[key][y_or_u]
                + N * roa
                + P * np.linalg.norm(action - self.INIT_ACTION[key])
                + 0.5 * (1 - abs(info[key]['p_set_p_max'])) ** 2
            )
            rewards[key] = r

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
            r = (
                -(
                    M * y ** 2
                    + N * np.sum((action - old_action) ** 2)
                    + P * np.sum((action - self.INIT_ACTION[key]) ** 2)
                )
                / 100
            )
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
            r += (
                -(
                    (
                        M * y ** 2
                        + N * np.sum((action - old_action) ** 2)
                        + P * np.sum((action - self.INIT_ACTION[key]) ** 2)
                    )
                )
                / 100
            )
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
        Logger = logger()

        M = self.k.sim_params['M']
        N = self.k.sim_params['N']
        P = self.k.sim_params['P']
        Q = self.k.sim_params['Q']
        T = self.k.sim_params['T']
        global_reward = 0
        # we accumulate agents reward into global_reward and divide it with the number of agents.
        y_or_u = 'u' if self.unbalance else 'y'

        component_y = 0
        component_oa = 0
        component_init = 0
        component_pset_pmax = 0
        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = self.INIT_ACTION[key]
            old_action = info[key]['old_action']
            if old_action is None:
                old_action = self.INIT_ACTION[key]

            r = 0

            action = np.array(action)
            old_action = np.array(old_action)
            if isinstance(self.action_space, Box):
                roa = np.abs(action - old_action).sum()
            elif (action == old_action).all():
                roa = 0
            else:
                roa = 1

            r += -(
                M * info[key][y_or_u]
                + N * roa
                + P * np.linalg.norm(action - self.INIT_ACTION[key])
                + Q * (1 - abs(info[key]['p_set_p_max'])) ** 2
            )

            component_y += -M * info[key][y_or_u]
            component_oa += -N * roa
            component_init += -P * np.linalg.norm(action - self.INIT_ACTION[key])
            component_pset_pmax += -Q * (1 - abs(info[key]['p_set_p_max'])) ** 2

            global_reward += r
        global_reward = global_reward / len(list(info.keys()))

        # voltage threshold on s701
        latest_va = self.k.node.nodes['s701a']['voltage'][self.k.time-self.k.sim_params['env_config']['sims_per_step']:self.k.time]
        latest_vb = self.k.node.nodes['s701b']['voltage'][self.k.time-self.k.sim_params['env_config']['sims_per_step']:self.k.time]
        latest_vc = self.k.node.nodes['s701c']['voltage'][self.k.time-self.k.sim_params['env_config']['sims_per_step']:self.k.time]

        global_reward -= T*np.sqrt(np.sum((latest_va - 1)**2 + (latest_vb - 1)**2 + (latest_vc - 1)**2))
        #v_pena = v_penb = v_penc = 0
        #if latest_va[latest_va < VOLTAGE_THRESHOLD_LB].size != 0:
        #    v_pena = np.mean((latest_va[latest_va < VOLTAGE_THRESHOLD_LB] - VOLTAGE_THRESHOLD_LB)**2)
        #if latest_vb[latest_vb < VOLTAGE_THRESHOLD_LB].size != 0:
        #    v_penb = np.mean((latest_vb[latest_vb < VOLTAGE_THRESHOLD_LB] - VOLTAGE_THRESHOLD_LB)**2)
        #if latest_vc[latest_vc < VOLTAGE_THRESHOLD_LB].size != 0:
        #    v_penc = np.mean((latest_vc[latest_vc < VOLTAGE_THRESHOLD_LB] - VOLTAGE_THRESHOLD_LB)**2)

        #global_reward -= T*np.mean([v_pena, v_penb, v_penc])


        n = len(list(info.keys()))
        for _ in range(self.env.k.sim_params['env_config']['sims_per_step']):
            Logger.log('component_reward', 'component_y', component_y / n)
            Logger.log('component_reward', 'component_oa', component_oa / n)
            Logger.log('component_reward', 'component_init', component_init / n)
            Logger.log('component_reward', 'component_pset_pmax', component_pset_pmax / n)

        return global_reward


###########################################################################
#                         MULTI-AGENT WRAPPER                             #
###########################################################################


class GroupRewardWrapper(RewardWrapper):
    def __init__(self, env, unbalance=False):
        super().__init__(env)
        self.unbalance = unbalance

    def reward(self, reward, info):
        re = {}
        re['defense_agent'] = np.mean([reward[key] for key in reward if 'adversary_' not in key])
        re['attack_agent'] = np.mean([reward[key] for key in reward if 'adversary_' in key])
        if np.isnan(re['defense_agent']):
            del re['defense_agent']
        if np.isnan(re['attack_agent']):
            del re['attack_agent']

        return re


class AdvLocalRewardWrapper(RewardWrapper):
    """
    The reward for each agent. It depends on agent local observation.
    """

    def __init__(self, env, unbalance=False):
        super().__init__(env)
        self.unbalance = unbalance

    def reward(self, reward, info):
        Logger = logger()
        M = self.k.sim_params['M']
        N = self.k.sim_params['N']
        P = self.k.sim_params['P']
        Q = self.k.sim_params['Q']
        rewards = {}
        y_or_u = 'u' if self.unbalance else 'y'
        # for each agent, we set the reward as under, note that agent reward is only a function of local information.
        component_y = 0
        component_oa = 0
        component_init = 0
        component_pset_pmax = 0
        count = 0

        adv_component_y = 0
        adv_component_oa = 0
        adv_count = 0

        for key in info.keys():
            action = info[key]['current_action']
            if action is None:
                action = self.INIT_ACTION[key]

            old_action = info[key]['old_action']
            if old_action is None:
                old_action = self.INIT_ACTION[key]

            action = np.array(action)
            old_action = np.array(old_action)
            if (action == old_action).all():
                roa = 0
            else:
                roa = 1

            if 'adversary_' not in key:
                count += 1
                r = -(
                    M * info[key][y_or_u]
                    + N * roa
                    + P * np.linalg.norm(action - self.INIT_ACTION[key])
                    + Q * (1 - abs(info[key]['p_set_p_max'])) ** 2
                )

                component_y += -M * info[key]['y']
                component_oa += -N * roa
                component_init += -P * np.linalg.norm(action - self.INIT_ACTION[key])
                component_pset_pmax += -Q * (1 - abs(info[key]['p_set_p_max'])) ** 2
            else:
                adv_count += 1
                r = M * info[key][y_or_u] - N * roa
                adv_component_y += M * info[key]['y']
                adv_component_oa += -N * roa
            rewards[key] = r

        for _ in range(self.env.k.sim_params['env_config']['sims_per_step']):
            Logger.log('component_reward', 'component_y', component_y / float(count))
            Logger.log('component_reward', 'component_oa', component_oa / float(count))
            Logger.log('component_reward', 'component_init', component_init / float(count))
            Logger.log('component_reward', 'component_pset_pmax', component_pset_pmax / float(count))

            if adv_count != 0:
                Logger.log('adv_component_reward', 'adv_component_y', adv_component_y / float(adv_count))
                Logger.log('adv_component_reward', 'adv_component_oa', adv_component_oa / float(adv_count))
            else:
                Logger.log('adv_component_reward', 'adv_component_y', 0)
                Logger.log('adv_component_reward', 'adv_component_oa', 0)

        return rewards

class PhaseSpecificRewardWrapper(RewardWrapper):
    def __init__(self, env, unbalance=False):
        super().__init__(env)
        self.unbalance = unbalance

    def reward(self, reward, info):
        Logger = logger()

        M = self.k.sim_params['M']
        N = self.k.sim_params['N']
        P = self.k.sim_params['P']
        Q = self.k.sim_params['Q']
        y_or_u = 'u' if self.unbalance else 'y'

        r = {}
        for key in info:
            action = info[key]['current_action']
            if action is None:
                action = self.INIT_ACTION[key]

            old_action = info[key]['old_action']
            if old_action is None:
                old_action = self.INIT_ACTION[key]

            action = np.array(action)
            old_action = np.array(old_action)
            if isinstance(self.action_space, Box):
                roa = np.abs(action - old_action).sum()
            elif (action == old_action).all():
                roa = 0
            else:
                roa = 1

            r[key] = -(
                M * info[key][y_or_u]
                + N * roa
                + P * np.linalg.norm(action - self.INIT_ACTION[key])
                + Q * (1 - abs(info[key]['p_set_p_max'])) ** 2
            )

        return r