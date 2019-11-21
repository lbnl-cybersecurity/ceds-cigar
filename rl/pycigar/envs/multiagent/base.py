import numpy as np
from gym.spaces import Box
from ray.rllib.env import MultiAgentEnv
from pycigar.envs.base import Env
from gym.spaces import Discrete


class MultiEnv(MultiAgentEnv, Env):

    @property
    def action_space(self):
        return Discrete(30)

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(5,), dtype=np.float32)

    def step(self, rl_actions):
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

        for _ in range(self.sim_params['env_config']['sims_per_step']):
            self.env_time += 1

            # perform action update for PV inverter device controlled by adaptive control
            if len(self.k.device.get_adaptive_device_ids()) > 0:
                adaptive_id = []
                control_setting = []
                for device_id in self.k.device.get_adaptive_device_ids():
                    adaptive_id.append(device_id)
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(adaptive_id, control_setting)

            # perform action update for PV inverter device controlled by fixed control
            if len(self.k.device.get_fixed_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_fixed_device_ids():
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_fixed_device_ids(), control_setting)

            # perform action update for PV inverter device controlled by RL control
            self.old_actions = {}
            if rl_actions is not None:
                for rl_id in rl_actions.keys():
                    self.old_actions[rl_id] = self.k.device.get_control_setting(rl_id)
            else:
                rl_actions = self.old_actions
            self.apply_rl_actions(rl_actions)
            self.additional_command()
            self.k.update(reset=False)

            # check whether the simulator sucessfully solved the powerflow
            converged = self.k.simulation.check_converged()
            if not converged:
                break

            states = self.get_state()
            next_observation = states

            # the episode will be finished if it is not converged.
            finish = not converged or (self.k.time == self.k.t)
            done = {}
            if finish:
                done['__all__'] = True
            else:
                done['__all__'] = False

            # we push the old action into additional info, which will return by the env after calling step.
            if self.old_actions != {}:
                infos = {key: {'old_action': self.old_actions[key]} for key in states.keys()}
            else:
                infos = {key: {'old_action': None} for key in states.keys()}

            # clip the action into a good range or not
            if self.sim_params['env_config']['clip_actions']:
                rl_clipped = self.clip_actions(rl_actions)
                reward = self.compute_reward(rl_clipped, fail=not converged)
            else:
                reward = self.compute_reward(rl_actions, fail=not converged)

            # Keep track of the devices which is in the list of tracking ids.
            if self.tracking_ids is not None:
                self.pycigar_tracking()

            return next_observation, reward, done, infos

    def get_old_actions(self):
        """
        function to get the old action.
        This will be deleted in the future. All the information needed to
        formulate new reward or observation in the wrappers should be saved in infos.
        """
        return self.old_actions

    def reset(self):
        self.old_actions = {}
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
            for key, action in rl_actions.items():
                rl_actions[key] = np.clip(
                    action,
                    a_min=self.action_space.low,
                    a_max=self.action_space.high)

        return rl_actions

    def apply_rl_actions(self, rl_actions=None):
        if rl_actions is None:
            return
        rl_clipped = self.clip_actions(rl_actions)
        self._apply_rl_actions(rl_clipped)

    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                action = actions
                self.k.device.apply_control(rl_id, action)

    def get_state(self):
        obs = {}

        for rl_id in self.k.device.get_rl_device_ids():
            connected_node = self.k.device.get_node_connected_to(rl_id)

            voltage = self.k.node.get_node_voltage(connected_node)
            solar_generation = self.k.device.get_solar_generation(rl_id)
            y = self.k.device.get_device_y(rl_id)

            p_inject = self.k.device.get_device_p_injection(rl_id)
            q_inject = self.k.device.get_device_q_injection(rl_id)

            observation = np.array([voltage, solar_generation, y, p_inject, q_inject])

            obs.update({rl_id: observation})

        return obs
