from pycigar.envs.base import Env
import numpy as np
from gym.spaces.box import Box
import numpy as numpy

A = 0
B = 100
C = 1
D = 0


class AdaptiveControlPVInverterEnv(Env):

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

    def get_state(self):
        obs = {}
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        rewards = {}
        for measure_id in self.tracking_ids:
            connected_node = self.k.device.get_node_connected_to(measure_id)
            voltage = self.k.node.get_node_voltage(connected_node)
            y = self.k.device.get_device_y(measure_id)
            p_inject = self.k.device.get_device_p_injection(measure_id)
            p_max = self.k.device.get_solar_generation(measure_id)
            r = -(np.sqrt(A*(1-voltage)**2 + B*y**2 + C*(1+p_inject/p_max)**2))
            rewards.update({measure_id: r})
        return rewards
