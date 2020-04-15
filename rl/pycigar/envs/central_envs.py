from pycigar.envs.rl_control_pv_inverter_env import RLControlPVInverterEnv
from pycigar.envs.wrappers import *
from pycigar.utils.logging import logger
import matplotlib.pyplot as plt


class CentralControlPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = RLControlPVInverterEnv(**kwargs)
        env = SingleRelativeInitDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env

    def plot(self, tracking_id):
        log_dict = logger().log_dict
        f, ax = plt.subplots(6, figsize=(25, 25))
        tracking_id = tracking_id
        node = self.env.unwrapped.k.device.get_node_connected_to(tracking_id)
        ax[0].set_title(tracking_id + " -- total reward: " + str(0))
        ax[0].plot(log_dict[node]['voltage'])
        print('voltage: {}'.format(len(log_dict[node]['voltage'])))
        ax[0].set_ylabel('voltage')
        ax[0].set_ylim((0.925, 1.07))
        # ax[0].axhline(1.03, color='r')
        # ax[0].axhline(0.95, color='r')
        ax[0].grid(b=True, which='both')
        ax[1].plot(log_dict[tracking_id]['y'])
        print('y: {}'.format(len(log_dict[tracking_id]['y'])))
        ax[1].set_ylabel('oscillation observer')
        ax[1].grid(b=True, which='both')
        ax[2].plot(log_dict[tracking_id]['q_set'])
        ax[2].plot(log_dict[tracking_id]['q_out'])
        print('q_out: {}'.format(len(log_dict[tracking_id]['q_out'])))
        ax[2].set_ylabel('reactive power')
        ax[2].grid(b=True, which='both')
        labels = ['a1', 'a2', 'a3', 'a4', 'a5']
        [a1, a2, a3, a4, a5] = ax[3].plot(log_dict[tracking_id]['control_setting'])
        print('a: {}'.format(len(log_dict[tracking_id]['control_setting'])))
        ax[3].set_ylabel('action')
        ax[3].grid(b=True, which='both')
        plt.legend([a1, a2, a3, a4, a5], labels, loc=1)

        save_path = '/home/toanngo/{}_{}_result_{}.png'.format(1, 1, 1)

        f.savefig(save_path)
        plt.close(f)
        return f


class NewCentralControlPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = RLControlPVInverterEnv(**kwargs)
        env = NewSingleRelativeInitDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = NewCentralLocalObservationWrapper(env)
        env = NewCentralFramestackObservationWrapper(env)
        self.env = env


class CentralControlPVInverterContinuousEnv(Wrapper):
    def __init__(self, **kwargs):
        env = RLControlPVInverterEnv(**kwargs)
        env = SingleRelativeInitContinuousActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalContinuousObservationWrapper(env)
        env = CentralFramestackContinuousObservationWrapper(env)
        self.env = env
