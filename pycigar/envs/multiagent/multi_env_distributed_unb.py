import numpy as np
from gym.spaces import Box
from ray.rllib.env import MultiAgentEnv
from pycigar.envs.base import Env
from pycigar.controllers import AdaptiveFixedController
import re
from pycigar.utils.logging import logger

class UnbMultiEnv(MultiAgentEnv, Env):
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
        randomize_rl_update = {}
        if rl_actions is None:
            rl_actions = self.old_actions

        for rl_id in rl_actions.keys():
            self.old_actions[rl_id] = self.k.device.get_control_setting(rl_id)
            randomize_rl_update[rl_id] = np.random.randint(low=0, high=3)

        # TODOs: disable defense action here
        # if rl_actions != {}:
        #    for key in rl_actions:
        #        if 'adversary_' not in key:
        #            rl_actions[key] = self.k.device.get_control_setting(key) #[1.014, 1.015, 1.015, 1.016, 1.017]

        for _ in range(self.sim_params['env_config']['sims_per_step']):
            self.env_time += 1
            # perform action update for PV inverter device controlled by RL control
            if rl_actions != {}:
                temp_rl_actions = {}
                for rl_id in self.k.device.get_rl_device_ids():
                    if rl_id in rl_actions:
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
                    print('not converged')
                    break

                if observations == {}:
                    observations = self.get_state()
                else:
                    new_state = self.get_state()
                    for device_name in new_state:
                        if device_name not in observations:
                            observations[device_name] = new_state[device_name]
                        for prop in new_state[device_name]:
                            if not isinstance(observations[device_name][prop], list):
                                observations[device_name][prop] = [observations[device_name][prop]]
                            else:
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

        for k, v in enumerate(obs):
            obs[v]['voltage'] = observations[v]['voltage'][-1]
            obs[v]['u'] = observations[v]['u'][-1] #disable if use current u

        # the episode will be finished if it is not converged.
        finish = not converged or (self.k.time == self.k.t)
        done = {}

        if finish:
            done['__all__'] = True
        else:
            done['__all__'] = False

        infos = {}

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
        Logger = logger()
        u_worst, v_worst, u_mean, u_std, v_all, u_all = self.k.kernel_api.get_worst_u_node_real()
        Logger.log('u_metrics', 'u_worst', u_worst)
        Logger.log('u_metrics', 'u_mean', u_mean)
        Logger.log('u_metrics', 'u_std', u_std)
        Logger.log('v_metrics', str(self.k.time), v_all)

        for rl_id in self.k.device.get_rl_device_ids():
            connected_node = self.k.device.get_node_connected_to(rl_id)
            bus = re.findall('\d+', connected_node)[0]

            obs.update(
                {
                    rl_id: {
                        'voltage': (np.array(v_all[bus])-1)*10*2,
                        'u': u_all[bus]/0.1,
                        'p_set_p_max': self.k.device.get_device_p_set_p_max(rl_id),
                        'p_set': self.k.device.get_device_p_set_relative(rl_id),
                        'sbar_solar_irr': self.k.device.get_device_sbar_solar_irr(rl_id)*1.5e-3,
                    }
                }
            )

        return obs
