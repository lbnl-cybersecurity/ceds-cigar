import numpy as np
from gym.spaces.box import Box

from pycigar.envs.base import Env


class CentralEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'),
                   shape=(5,), dtype=np.float64)

    @property
    def action_space(self):
        return Box(low=0.5, high=1.5, shape=(5,), dtype=np.float64)

    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                action = actions
                self.k.device.apply_control(rl_id, action)

    def step(self, rl_actions):
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
        next_observation = None
        self.old_actions = {}
        for rl_id in self.k.device.get_rl_device_ids():
            self.old_actions[rl_id] = self.k.device.get_control_setting(rl_id)
        randomize_rl_update = {}
        if rl_actions is not None:
            for rl_id in self.k.device.get_rl_device_ids():
                randomize_rl_update[rl_id] = np.random.randint(low=0, high=3)
        else:
            rl_actions = self.old_actions

        count = 0

        for _ in range(self.sim_params['env_config']["sims_per_step"]):
            self.env_time += 1
            # perform action update for PV inverter device controlled by RL control
            temp_rl_actions = {}
            for rl_id in self.k.device.get_rl_device_ids():
                temp_rl_actions[rl_id] = rl_actions[rl_id]
            rl_dict = {}
            for rl_id in temp_rl_actions.keys():
                if randomize_rl_update[rl_id] == 0:
                    rl_dict[rl_id] = temp_rl_actions[rl_id]
                else:
                    randomize_rl_update[rl_id] -= 1

            for rl_id in rl_dict.keys():
                del temp_rl_actions[rl_id]

            self.apply_rl_actions(rl_dict)

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

            self.additional_command()

            if self.k.time <= self.k.t:
                self.k.update(reset=False)

                # check whether the simulator sucessfully solved the powerflow
                converged = self.k.simulation.check_converged()
                if not converged:
                    break

                states = self.get_state()
                if next_observation is None:
                    next_observation = states
                else:
                    next_observation += states
                count += 1

            if self.k.time >= self.k.t:
                break

        next_observation = next_observation / count

        # the episode will be finished if it is not converged.
        finish = not converged or (self.k.time == self.k.t)

        if finish:
            done = True
        else:
            done = False

        infos = {key: {'voltage': self.k.node.get_node_voltage(self.k.device.get_node_connected_to(key)),
                       'y': next_observation[2],
                       'p_inject': self.k.device.get_device_p_injection(key),
                       'env_time': self.env_time,
                       'p_set': next_observation[4],
                       'p_set_p_max': next_observation[3],
                       } for key in self.k.device.get_rl_device_ids()}

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

        return next_observation, reward, done, infos

    def get_state(self):
        obs = {}
        for rl_id in self.k.device.get_rl_device_ids():
            connected_node = self.k.device.get_node_connected_to(rl_id)
            voltage = self.k.node.get_node_voltage(connected_node)
            solar_generation = self.k.device.get_solar_generation(rl_id)
            y = self.k.device.get_device_y(rl_id)
            p_set_p_max = self.k.device.get_device_p_set_p_max(rl_id)
            p_set = self.k.device.get_device_p_set_relative(rl_id)
            observation = np.array([voltage, solar_generation, y, p_set_p_max, p_set])
            obs.update({rl_id: observation})

        obs = np.mean(np.array(list(obs.values())), axis=0)

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        return 0
