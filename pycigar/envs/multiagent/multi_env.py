import numpy as np
from gym.spaces import Box
from ray.rllib.env import MultiAgentEnv
from pycigar.envs.base import Env
from pycigar.controllers import AdaptiveFixedController
from pycigar.utils.logging import logger

class MultiEnv(MultiAgentEnv, Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def action_space(self):
        return Box(low=0.5, high=1.5, shape=(5,), dtype=np.float64)

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float64)

    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                action = actions
                self.k.device.apply_control(rl_id, action)

    def step(self, rl_actions, randomize_rl_update=None):
        """Perform 1 step forward in the environment.

        Parameters
        ----------
        rl_actions : dict
            A dictionary of actions of all the rl agents.

        Returns
        -------
        Tuple
            A tuple of (obs, reward, done, infos).
            obs: a dictionary of new observation from the environment.
            reward: a dictionary of reward received by agents.
            done: a dictionary of done of each agent. Each agent can be done before the environment actually done.
                  {'id_1': False, 'id_2': False, '__all__': False}
                  a simulation is delared done when '__all__' key has the value of True,
                  indicate all agents has finished their job.

        """
        observations = {}
        self.old_actions = {}
        for rl_id in self.k.device.get_rl_device_ids():
            self.old_actions[rl_id] = self.k.device.get_control_setting(rl_id)
        if rl_actions is None:
            rl_actions = self.old_actions

        if randomize_rl_update is None:
            randomize_rl_update = np.random.randint(5, size=len(self.k.device.get_rl_device_ids()))


        # TODOs: disable defense action here
        # if rl_actions != {}:
        #    for key in rl_actions:
        #        if 'adversary_' not in key:
        #            rl_actions[key] = self.k.device.get_control_setting(key) #[1.014, 1.015, 1.015, 1.016, 1.017]

        for _ in range(self.sim_params['env_config']['sims_per_step']):
            self.env_time += 1
            rl_ids_key = np.array(self.k.device.get_rl_device_ids())[randomize_rl_update == 0]
            randomize_rl_update -= 1
            rl_dict = {k: rl_actions[k] for k in rl_ids_key}
            self.apply_rl_actions(rl_dict)

            # perform action update for PV inverter device controlled by adaptive control
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

                if observations == {}:
                    new_state = self.get_state()
                    for device_name in new_state:
                        if device_name not in observations:
                            observations[device_name] = {}
                        for prop in new_state[device_name]:
                            observations[device_name][prop] = [new_state[device_name][prop]]
                else:
                    new_state = self.get_state()
                    for device_name in new_state:
                        if device_name not in observations:
                            observations[device_name] = new_state[device_name]
                        for prop in new_state[device_name]:
                            observations[device_name][prop].append(new_state[device_name][prop])

            if self.k.time >= self.k.t:
                break

        list_device = self.k.device.get_rl_device_ids()
        list_device_observation = list(observations.keys())
        for device in list_device_observation:
            if device not in list_device:
                del observations[device]
        obs = {
            device: {prop: np.mean(observations[device][prop]) for prop in observations[device]}
            for device in observations
        }

        for device in obs:
            obs[device]['v'] = observations[device]['v'][-1]
            obs[device]['u'] = observations[device]['u'][-1]
            
        # the episode will be finished if it is not converged.
        finish = not converged or (self.k.time == self.k.t)
        done = {}
        if finish:
            done['__all__'] = True
        else:
            done['__all__'] = False

        infos = {
            key: {
                'v': obs[key]['v'],
                'y': obs[key]['y'],
                'u': obs[key]['u'],
                'p_set_p_max': obs[key]['p_set_p_max'],
                'sbar_solar_irr': obs[key]['sbar_solar_irr'],
            }
            for key in self.k.device.get_rl_device_ids()
        }

        for key in self.k.device.get_rl_device_ids():
            if self.old_actions != {}:
                if key in self.old_actions:
                    infos[key]['old_action'] = self.old_actions[key]
                else:
                    infos[key]['old_action'] = self.k.device.get_control_setting(key)
            else:
                infos[key]['old_action'] = None

            if rl_actions != {}:
                if key in rl_actions:
                    infos[key]['current_action'] = rl_actions[key]
                else:
                    infos[key]['current_action'] = self.k.device.get_control_setting(key)
            else:
                infos[key]['current_action'] = None

        # clip the action into a good range or not
        if self.sim_params['env_config']['clip_actions']:
            rl_clipped = self.clip_actions(rl_actions)
            reward = self.compute_reward(rl_clipped, fail=not converged)
        else:
            reward = self.compute_reward(rl_actions, fail=not converged)

        return obs, reward, done, infos

    def reset(self):
        # TODOs: delete here
        # self.tempo_controllers = {}
        self.accumulate_current = {}
        self.average_current = {}
        self.average_current_done = False
        self.env_time = 0
        self.k.update(reset=True)  # hotfix: return new sim_params sample in kernel?
        self.sim_params = self.k.sim_params
        states = self.get_state()

        self.INIT_ACTION = {}
        pv_device_ids = self.k.device.get_pv_device_ids()
        for device_id in pv_device_ids:
            self.INIT_ACTION[device_id] = np.array(self.k.device.get_control_setting(device_id))
        return states

    def get_state(self):
        obs = {}
        u_worst, _, _, _, v_all, u_all, load_to_bus = self.k.kernel_api.get_worst_u_node()
        Logger = logger()

        if not self.sim_params['vectorized_mode']:
            for rl_id in self.k.device.get_rl_device_ids():
                connected_node = self.k.device.get_node_connected_to(rl_id)
                obs.update(
                    {
                        rl_id: {
                            'v': v_all[load_to_bus[connected_node]],
                            'u': u_all[load_to_bus[connected_node]],
                            'y': self.k.device.get_device_y(rl_id),
                            'p_set_p_max': self.k.device.get_device_p_set_p_max(rl_id),
                            'sbar_solar_irr': self.k.device.get_device_sbar_solar_irr(rl_id)
                        }
                    }
                )
                Logger.log(rl_id, 'u', u_all[load_to_bus[connected_node]])
                Logger.log(rl_id, 'v', v_all[load_to_bus[connected_node]])
        else:
            y = self.k.device.get_vectorized_y()
            p_set_p_max = self.k.device.get_vectorized_device_p_set_p_max()
            sbar_solar_irr = self.k.device.get_vectorized_device_sbar_solar_irr()
            for i, rl_id in enumerate(self.k.device.get_rl_device_ids()):
                connected_node = self.k.device.get_node_connected_to(rl_id)
                idx = self.k.device.vectorized_pv_inverter_device.list_device.index(rl_id)
                obs.update(
                    {
                        rl_id: {
                            'v': v_all[load_to_bus[connected_node]],
                            'u': u_all[load_to_bus[connected_node]],
                            'y': y[idx],
                            'p_set_p_max': p_set_p_max[idx],
                            'sbar_solar_irr': sbar_solar_irr[idx],
                         }
                    }
                )
                Logger.log(rl_id, 'u', u_all[load_to_bus[connected_node]])
                Logger.log(rl_id, 'v', v_all[load_to_bus[connected_node]])
            Logger.log('u_metrics', 'u_worst', u_worst)
            Logger.log('y_metrics', 'y_worst', max(y))
        return obs

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
                    self.relative_line_current[line] = self.line_current[line] #/self.average_current[line]

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