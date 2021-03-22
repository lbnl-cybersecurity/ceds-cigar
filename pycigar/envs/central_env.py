import numpy as np
from gym.spaces.box import Box

from pycigar.envs.base import Env
from pycigar.utils.logging import logger
from copy import deepcopy
import time
import re
import opendssdirect as dss

class CentralEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32)

    @property
    def action_space(self):
        return Box(low=0.5, high=1.5, shape=(5,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            self.k.device.apply_control(list(rl_actions.keys()), list(rl_actions.values()))

    def step(self, rl_actions, randomize_rl_update=None):

        """Move the environment one step forward.

        Parameters
        ----------
        rl_actions : dict
            A dictionary of actions of each agents controlled by RL algorithms.

        Returns
        -------
        Tuple
            A tuple of (obs, reward, done, infos).
            obs: a dictionary of new observation from the environment.
            reward: a dictionary of reward received by agents.
            done: bool
        """

        observations = []
        self.old_actions = {}
        for rl_id in self.k.device.get_rl_device_ids():
            self.old_actions[rl_id] = self.k.device.get_control_setting(rl_id)

        # need to refactor this bulk
        if randomize_rl_update is None:
            randomize_rl_update = np.random.randint(5, size=len(self.k.device.get_rl_device_ids()))

        if rl_actions is None:
            rl_actions = self.old_actions

        for _ in range(self.sim_params['env_config']["sims_per_step"]):
            self.env_time += 1
            rl_ids_key = np.array(self.k.device.get_rl_device_ids())[randomize_rl_update == 0]
            randomize_rl_update -= 1
            rl_dict = {k: rl_actions[k] for k in rl_ids_key}
            self.apply_rl_actions(rl_dict)

            # perform action update for PV inverter device
            if len(self.k.device.get_norl_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_norl_device_ids():
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_norl_device_ids(), control_setting)

            self.additional_command()
            if self.k.time <= self.k.t:
                self.k.update(reset=False)
                # check whether the simulator sucessfully solved the powerflow
                converged = self.k.simulation.check_converged()
                if not converged:
                    break

                observations.append(self.get_state())
            if self.k.time >= self.k.t:
                break

        try:
            obs = {k: np.mean([d[k] for d in observations]) for k in observations[0]}
            obs['v_worst'] = observations[-1]['v_worst']
            obs['u_worst'] = observations[-1]['u_worst']
            obs['y_worst'] = observations[-1]['y_worst']
            obs['u_mean'] = obs['u_mean']
            obs['y_mean'] = obs['y']
        except IndexError:
            obs = {'p_set_p_max': 0.0, 'sbar_solar_irr': 0.0, 'y': 0.0}
            obs['v_worst'] = [0, 0, 0]
            obs['u_worst'] = 0
            obs['u_mean'] = 0
            obs['y_worst'] = 0
            obs['y_mean'] = 0

        # the episode will be finished if it is not converged.
        done = not converged or (self.k.time == self.k.t)
        if 'protection' in self.sim_params:
            protection = (np.array(list(self.relative_line_current.values())).T/self.sim_params['protection']['threshold']-1).T
            infos = {
                key: {
                    'protection': protection.flatten(),
                    'y_worst': obs['y_worst'],
                    'u_worst': obs['u_worst'],
                    'v_worst': obs['v_worst'],
                    'u_mean': obs['u_mean'],
                    'y_mean': obs['y_mean'],
                    'y': obs['y'],
                    'p_set_p_max': obs['p_set_p_max'],
                    'sbar_solar_irr': obs['sbar_solar_irr'],
                }
                for key in self.k.device.get_rl_device_ids()
            }
        else:
            infos = {
                key: {
                    'y_worst': obs['y_worst'],
                    'u_worst': obs['u_worst'],
                    'v_worst': obs['v_worst'],
                    'u_mean': obs['u_mean'],
                    'y_mean': obs['y_mean'],
                    'y': obs['y'],
                    'p_set_p_max': obs['p_set_p_max'],
                    'sbar_solar_irr': obs['sbar_solar_irr'],
                }
                for key in self.k.device.get_rl_device_ids()
            }

        for key in self.k.device.get_rl_device_ids():
            if self.old_actions is not None:
                infos[key]['old_action'] = self.old_actions[key]
            else:
                infos[key]['old_action'] = None
            if rl_actions is not None:
                infos[key]['current_action'] = rl_actions[key]
            else:
                infos[key]['current_action'] = None

        # clip the action into a good range or not
        if self.sim_params['env_config']['clip_actions']:
            rl_clipped = self.clip_actions(rl_actions)
            reward = self.compute_reward(rl_clipped, fail=not converged)
        else:
            reward = self.compute_reward(rl_actions, fail=not converged)

        return obs, reward, done, infos

    def get_state(self):
        obs = []
        Logger = logger()
        u_worst, v_worst, u_mean, u_std, v_all, u_all_bus, self.load_to_bus = self.k.kernel_api.get_worst_u_node()

        Logger.log('u_metrics', 'u_worst', u_worst)
        Logger.log('u_metrics', 'u_mean', u_mean)
        Logger.log('u_metrics', 'u_std', u_std)
        Logger.log('v_metrics', str(self.k.time), v_all)

        y_worst = 0

        if not self.sim_params['vectorized_mode']:
            for rl_id in self.k.device.get_rl_device_ids():
                y = self.k.device.get_device_y(rl_id)
                obs.append(
                    {
                        'y': y,
                        'p_set_p_max': self.k.device.get_device_p_set_p_max(rl_id),
                        'sbar_solar_irr': self.k.device.get_device_sbar_solar_irr(rl_id)
                    }
                )
                if y_worst < y:
                    y_worst = y

            if obs:
                result = {k: np.mean([d[k] for d in obs]) for k in obs[0]}
                result['y_worst'] = y_worst
                result['v_worst'] = v_worst
                result['u_worst'] = u_worst
                result['u_mean'] = u_mean
                result['u_std'] = u_std
                return result
            else:
                return {}
        else:
            result = {}
            try:
                y = self.k.device.get_vectorized_y()
                y_worst_node_idx = y.argmax()
                y_worst_node = self.k.device.vectorized_pv_inverter_device.list_device[y_worst_node_idx].split('_')[1]
                y_worst = np.max(y)
                y = np.mean(y)
                p_set_p_max = np.mean(self.k.device.get_vectorized_device_p_set_p_max()[0::2])
                sbar_solar_irr = np.mean(self.k.device.get_vectorized_device_sbar_solar_irr()[0::2])
                result = {'y': y,
                          'p_set_p_max': p_set_p_max,
                          'sbar_solar_irr': sbar_solar_irr,
                         }
                result['y_worst'] = y_worst
                result['v_worst'] = v_worst
                Logger.log('v_worst_metrics', 'v_worst', v_worst)
                Logger.log('y_metrics', 'y_worst', y_worst)
                Logger.log('y_metrics', 'y_worst_node', y_worst_node)
                Logger.log('y_metrics', 'y_mean', y)
                result['u_worst'] = u_worst
                result['u_mean'] = u_mean
                result['u_std'] = u_std
                return result
            except:
                return result

    def compute_reward(self, rl_actions, **kwargs):
        return 0

    def additional_command(self):
        current = self.k.kernel_api.get_all_currents()

        if 'protection' in self.sim_params:
            self.line_current = {}
            self.relative_line_current = {}
            for line in self.sim_params['protection']['line']:
                self.line_current[line] = np.array(current[line])
                if line not in self.accumulate_current:
                    self.accumulate_current[line] = [current[line]]
                elif line in self.accumulate_current and line not in self.average_current:
                    self.accumulate_current[line].append(current[line])
                if len(self.accumulate_current[line]) == self.sim_params['env_config']['sims_per_step'] and line not in self.average_current:
                    self.average_current[line] = np.mean(self.accumulate_current[line], axis=0)
                    self.average_current_done = True

            if self.average_current_done:
                for line in self.sim_params['protection']['line']:
                    self.relative_line_current[line] = self.line_current[line]

        if not self.sim_params['is_disable_log']:
            Logger = logger()
            for line in current:
                Logger.log('current', line, current[line])

            if 'protection' in self.sim_params:
                if self.average_current_done:
                    for line in self.relative_line_current:
                        Logger.log('relative_current', line, self.relative_line_current[line])
                else:
                    for line in self.sim_params['protection']['line']:
                        Logger.log('relative_current', line, np.ones(3))


            capacitors = dss.Capacitors.AllNames()
            for cap in capacitors:
                dss.Capacitors.Name(cap)
                Logger.log('capbank', cap, dss.Capacitors.States()[0])

