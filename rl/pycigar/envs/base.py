import traceback
import numpy as np
import atexit
import gym
from gym.spaces import Box
from pycigar.core.kernel.kernel import Kernel
import matplotlib.pyplot as plt
from datetime import datetime
import pycigar.config as config
import os


class Env(gym.Env):

    def __init__(self, sim_params, simulator='opendss', tracking_ids=None):
        self.sim_params = sim_params
        self.state = None
        self.simulator = simulator

        self.k = Kernel(simulator=self.simulator,
                        sim_params=sim_params)

        kernel_api = self.k.simulation.start_simulation()
        self.k.pass_api(kernel_api)
        self.k.scenario.start_scenario()

        atexit.register(self.terminate)

        # tracking
        self.tracking_ids = tracking_ids

    def restart_simulation(self, sim_params, render=None):
        pass

    def setup_initial_state(self):
        pass

    def step(self, rl_actions):
        for _ in range(self.sim_params['env_config']["sims_per_step"]):
            self.env_time += 1

            # perform action update for PV inverter device
            if len(self.k.device.get_adaptive_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_adaptive_device_ids():
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_adaptive_device_ids(), control_setting)

            # perform action update for PV inverter device
            if len(self.k.device.get_fixed_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_fixed_device_ids():
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_fixed_device_ids(), control_setting)

            # perform action update for RL inverter device
            self.apply_rl_actions(rl_actions)
            self.additional_command()
            self.k.update(reset=False)

            converged = self.k.simulation.check_converged()
            if not converged:
                break

            states = self.get_state()
            next_observation = states
            done = not converged or (self.k.time == self.k.t)

            infos = {}

            if self.sim_params['env_config']['clip_actions']:
                rl_clipped = self.clip_actions(rl_actions)
                reward = self.compute_reward(rl_clipped, fail=not converged)
            else:
                reward = self.compute_reward(rl_actions, fail=not converged)

            # tracking
            if self.tracking_ids is not None:
                self.pycigar_tracking()

            #if done and self.tracking_ids is not None:
            #    self.plot()

            return next_observation, reward, done, infos

    def reset(self):
        self.env_time = 0
        self.k.update(reset=True)
        states = self.get_state()
        # tracking
        if self.tracking_ids is not None:
            self.pycigar_tracking()

        return states

    def additional_command(self):
        pass

    def clip_actions(self, rl_actions=None):
        if rl_actions is None:
            return None

        if isinstance(self.action_space, Box):
            rl_actions = np.clip(
                rl_actions,
                a_min=self.action_space.low,
                a_max=self.action_space.high)
        return rl_actions

    def apply_rl_actions(self, rl_actions=None):
        if rl_actions is None:
            return None

        rl_clipped = self.clip_actions(rl_actions)
        self._apply_rl_actions(rl_clipped)

    def _apply_rl_actions(self, rl_actions):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        return NotImplementedError

    def compute_reward(self, rl_actions, **kwargs):
        return 0

    def terminate(self):
        try:
            # close everything within the kernel
            self.k.close()
        except FileNotFoundError:
            print(traceback.format_exc())

    def pycigar_tracking(self):
        if self.env_time == 0:
            self.tracking_infos = {}
            for tracking_id in self.tracking_ids:
                self.tracking_infos[tracking_id] = dict(v_val=[],
                                                        y_val=[],
                                                        p_val=[],
                                                        a_val=[])

        for tracking_id in self.tracking_ids:
            node_id = self.k.device.get_node_connected_to(tracking_id)
            self.tracking_infos[tracking_id]['v_val'].append(self.k.node.get_node_voltage(node_id))
            self.tracking_infos[tracking_id]['y_val'].append(self.k.device.get_device_y(tracking_id))
            p_max = self.k.device.get_solar_generation(tracking_id)
            p_inject = self.k.device.get_device_p_injection(tracking_id)
            self.tracking_infos[tracking_id]['p_val'].append((1+p_inject/p_max)**2)
            self.tracking_infos[tracking_id]['a_val'].append(list(self.k.device.get_control_setting(tracking_id)))

    def plot(self, exp_tag='', env_name='', iteration=0):
        num_col = len(self.tracking_infos.keys())
        f, ax = plt.subplots(4, num_col, figsize=(25, 20))
        for col in range(num_col):
            tracking_id = list(self.tracking_infos.keys())[col]
            ax[0, col].set_title(tracking_id)
            ax[0, col].plot(self.tracking_infos[tracking_id]['v_val'])
            ax[0, col].set_ylabel('voltage')
            ax[1, col].plot(self.tracking_infos[tracking_id]['y_val'])
            ax[1, col].set_ylabel('oscillation observer')
            ax[2, col].plot(self.tracking_infos[tracking_id]['p_val'])
            ax[2, col].set_ylabel('(1 + p_inject/p_max)**2')
            labels = ['a1', 'a2', 'a3', 'a4', 'a5']
            [a1, a2, a3, a4, a5] = ax[3, col].plot(self.tracking_infos[tracking_id]['a_val'])
            ax[3, col].set_ylabel('action')
            plt.legend([a1, a2, a3, a4, a5], labels, loc=1)

        if not os.path.exists(os.path.join(config.LOG_DIR, exp_tag)):
            os.makedirs(os.path.join(config.LOG_DIR, exp_tag))
        save_path = os.path.join(os.path.join(config.LOG_DIR, exp_tag), '{}_{}_result_{}.png'.format(env_name, iteration, datetime.now().strftime("%H:%M:%S.%f_%d-%m-%Y")))

        f.savefig(save_path)
        plt.close(f)
