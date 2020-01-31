from pycigar.envs.central_base import CentralEnv
import numpy as np
from gym.spaces.box import Box
import numpy as numpy

NUM_FRAMES = 5

class RLControlPVInverterEnv(CentralEnv):

    @property
    def observation_space(self):
        self.num_frames = NUM_FRAMES

        return Box(low=-float('inf'), high=float('inf'),
                   shape=(5, ), dtype=np.float32)

    @property
    def action_space(self):
        return Box(low=0.5, high=1.5, shape=(5,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                action = actions
                self.k.device.apply_control(rl_id, action)

    def get_state(self):
        obs = {}
        sample_id = self.k.device.get_rl_device_ids()[0]
        for rl_id in self.k.device.get_rl_device_ids():
            connected_node = self.k.device.get_node_connected_to(rl_id)

            voltage = self.k.node.get_node_voltage(connected_node)
            solar_generation = self.k.device.get_solar_generation(rl_id)
            y = self.k.device.get_device_y(rl_id)

            p_inject = self.k.device.get_device_p_injection(rl_id)
            q_inject = self.k.device.get_device_q_injection(rl_id)

            observation = np.array([voltage, solar_generation, y, p_inject, q_inject])

            obs.update({rl_id: observation})
        return obs[sample_id]

    def compute_reward(self, rl_actions, **kwargs):
        return 0
