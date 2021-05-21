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

    def __init__(self, env, unbalance=False, multi_attack=False):
        super().__init__(env)
        self.unbalance = unbalance
        self.multi_attack = multi_attack

    def reward(self, reward, info):
        Logger = logger()

        M = self.k.sim_params['M']
        N = self.k.sim_params['N']
        P = self.k.sim_params['P']
        Q = self.k.sim_params['Q']
        T = self.k.sim_params['T']
        Z = self.k.sim_params['Z']

        global_reward = 0
        # we accumulate agents reward into global_reward and divide it with the number of agents.
        y_or_u = 'u_worst' if self.unbalance else 'y'

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
            """if info[key]['no_action_penalty']:
                roa = 1
            else:
                roa = 0"""

            if not self.multi_attack:
                r += -(
                    M * info[key][y_or_u]
                    + N * roa
                    + P * np.linalg.norm(action - self.INIT_ACTION[key])
                    + Q * (1 - abs(info[key]['p_set_p_max'])) ** 2
                )
            else:
                r += -(
                      T * info[key]['u_worst']
                    + M * info[key]['y']
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

        if 'protection' in info[key]:
            protection_penalty = max(max(info[key]['protection']), 0)
            global_reward -= protection_penalty * Z

        if global_reward == float('inf') or global_reward == float('-inf') or global_reward == float('nan') or np.isnan(global_reward):
            print('check this out')
            global_reward = -100
        #print(global_reward, type(global_reward), global_reward == 'nan', np.isnan(global_reward))

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


###############################################################################
###############################################################################
class ClusterRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cluster = env.k.sim_params['cluster']
        self.env = env
        self.env.unwrapped.total_reward = {'1': np.zeros(5),'2': np.zeros(5),'3': np.zeros(5),'4': np.zeros(5), '5': np.zeros(5)}

    def reward(self, reward, info):
        M = self.k.sim_params['M']
        N = self.k.sim_params['N']
        P = self.k.sim_params['P']
        Q = self.k.sim_params['Q']
        T = self.k.sim_params['T']

        cluster_reward = {}
        for agent in self.cluster.keys():
            reward = 0
            r_T = 0
            r_M = 0
            r_N = 0
            r_P = 0
            r_Q = 0

            worst_y = 0
            worst_u = 0
            for key in self.cluster[agent]:
                key = 'inverter_' + key
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

                r = -(
                    #T * info[key]['u']
                    #+ M * info[key]['y']
                    + N * roa
                    + P * np.linalg.norm(action - self.INIT_ACTION[key])
                    + Q * (1 - abs(info[key]['p_set_p_max'])) ** 2
                )
                r_T += - T * info[key]['u']
                r_M += - M * info[key]['y']
                r_N += - N * roa
                r_P += - P * np.linalg.norm(action - self.INIT_ACTION[key])
                r_Q += - Q * (1 - abs(info[key]['p_set_p_max'])) ** 2

                reward += r
                if worst_y < info[key]['y']:
                    worst_y = info[key]['y']
                if worst_u < info[key]['u']:
                    worst_u = info[key]['u']
            reward = reward / len(self.cluster[agent])

            r_N = r_N / len(self.cluster[agent])
            r_P = r_P / len(self.cluster[agent])
            r_Q = r_Q / len(self.cluster[agent])
            #print('a: {}, r_M: {}, r_N: {}, r_P: {}, r_Q: {}, r_T: {}'.format(agent, r_M, r_N, r_P, r_Q, r_T))
            r_T = - T * worst_u
            r_M = - M * worst_y
            reward += r_T
            reward += r_M
            self.env.unwrapped.total_reward[agent] += np.array([r_M, r_N, r_P, r_Q, r_T])
            cluster_reward[agent] = reward
            if not self.k.sim_params['is_disable_log']:
                Logger = logger()
                for _ in range(self.env.k.sim_params['env_config']['sims_per_step']):
                    Logger.log('reward_{}'.format(agent), 'M', r_M)
                    Logger.log('reward_{}'.format(agent), 'N', r_N)
                    Logger.log('reward_{}'.format(agent), 'P', r_P)
                    Logger.log('reward_{}'.format(agent), 'Q', r_Q)
                    Logger.log('reward_{}'.format(agent), 'T', r_T)

        return cluster_reward
