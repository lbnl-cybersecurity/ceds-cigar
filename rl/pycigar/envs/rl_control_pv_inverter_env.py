import numpy as np
from gym.spaces.box import Box

from pycigar.envs.central_base import CentralEnv

NUM_FRAMES = 5


class RLControlPVInverterEnv(CentralEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frames = NUM_FRAMES

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
