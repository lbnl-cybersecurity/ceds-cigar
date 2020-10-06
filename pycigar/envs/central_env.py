import numpy as np
from gym.spaces.box import Box

from pycigar.envs.base import Env
from pycigar.utils.logging import logger
from copy import deepcopy
import time
import re

class CentralEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float64)

    @property
    def action_space(self):
        return Box(low=0.5, high=1.5, shape=(5,), dtype=np.float64)

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

            Logger = logger()
            if 'randomize_rl_update' not in Logger.custom_metrics:
                Logger.custom_metrics['randomize_rl_update'] = [deepcopy(randomize_rl_update)]
            else:
                Logger.custom_metrics['randomize_rl_update'].append(deepcopy(randomize_rl_update))

        if rl_actions is None:
            rl_actions = self.old_actions

        for _ in range(self.sim_params['env_config']["sims_per_step"]):
            start_time = time.time()
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

            if 'step_only_time' not in logger().custom_metrics:
                logger().custom_metrics['step_only_time'] = 0
            logger().custom_metrics['step_only_time'] += time.time() - start_time

        try:
            obs = {k: np.mean([d[k] for d in observations]) for k in observations[0]}
            obs['v_worst'] = observations[-1]['v_worst']
            obs['u_worst'] = observations[-1]['u_worst']
        except IndexError:
            obs = {'p_set': 0.0, 'p_set_p_max': 0.0, 'sbar_solar_irr': 0.0, 'solar_generation': 0.0, 'u': 0.0, 'voltage': 0.0, 'y': 0.0}
            obs['v_worst'] = [0, 0, 0]
            obs['u_worst'] = 0
            obs['u_mean'] = 0

        # the episode will be finished if it is not converged.
        done = not converged or (self.k.time == self.k.t)

        infos = {
            key: {
                'v_worst_all': np.array([k['v_worst'] for k in observations]),
                'u_worst': obs['u_worst'],
                'v_worst': obs['v_worst'],
                'u_mean': obs['u_mean'],
                'voltage': self.k.node.get_node_voltage(self.k.device.get_node_connected_to(key)),
                'y': obs['y'],
                'u': obs['u'],
                'p_inject': self.k.device.get_device_p_injection(key),
                'p_max': self.k.device.get_device_p_injection(key),
                'env_time': self.env_time,
                'p_set': obs['p_set'],
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
        u_worst, v_worst, u_mean, u_std, v_all, u_all_bus = self.k.kernel_api.get_worst_u_node()
        Logger.log('u_metrics', 'u_worst', u_worst)
        Logger.log('u_metrics', 'u_mean', u_mean)
        Logger.log('u_metrics', 'u_std', u_std)
        Logger.log('v_metrics', str(self.k.time), v_all)
        for rl_id in self.k.device.get_rl_device_ids():
            connected_node = self.k.device.get_node_connected_to(rl_id)
            bus = re.findall('\d+', connected_node)[0]
            obs.append(
                {
                    'u_worst': u_worst,
                    'u_mean': u_mean,
                    'u_std': u_std,
                    'voltage': self.k.node.get_node_voltage(connected_node),
                    'solar_generation': self.k.device.get_solar_generation(rl_id),
                    'y': self.k.device.get_device_y(rl_id),
                    'u': u_all_bus[bus],
                    'p_set_p_max': self.k.device.get_device_p_set_p_max(rl_id),
                    'sbar_solar_irr': self.k.device.get_device_sbar_solar_irr(rl_id),
                    'p_set': self.k.device.get_device_p_set_relative(rl_id),
                }
            )

        if obs:
            result = {k: np.mean([d[k] for d in obs]) for k in obs[0]}
            result['v_worst'] = v_worst
            return result
        else:
            return {}

    def compute_reward(self, rl_actions, **kwargs):
        return 0
