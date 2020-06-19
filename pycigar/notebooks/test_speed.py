"""
from pycigar.envs import Env
import yaml
import time
from pycigar.utils.logging import logger


class FooEnv(Env):
    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'),
                   shape=(5,), dtype=np.float64)

    @property
    def action_space(self):
        return Box(low=0.5, high=1.5, shape=(5,), dtype=np.float64)

    def step(self, rl_actions=None, randomize_rl_update=None):

        for _ in range(self.sim_params['env_config']["sims_per_step"]):
            self.env_time += 1

            # perform action update for PV inverter device
            if len(self.k.device.get_adaptive_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_adaptive_device_ids():
                    if 'update_controller_time' not in logger().custom_metrics.keys():
                        logger().custom_metrics['update_controller_time'] = 0
                    start_time = time.time()
                    action = self.k.device.get_controller(device_id).get_action(self)
                    logger().custom_metrics['update_controller_time'] += time.time() - start_time
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_adaptive_device_ids(), control_setting)

            # perform action update for PV inverter device
            if len(self.k.device.get_fixed_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_fixed_device_ids():
                    if 'update_controller_time' not in logger().custom_metrics.keys():
                        logger().custom_metrics['update_controller_time'] = 0
                    start_time = time.time()
                    action = self.k.device.get_controller(device_id).get_action(self)
                    logger().custom_metrics['update_controller_time'] += time.time() - start_time
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_fixed_device_ids(), control_setting)

            self.additional_command()

            if self.k.time <= self.k.t:
                self.k.update(reset=False)

                # check whether the simulator sucessfully solved the powerflow
                converged = self.k.simulation.check_converged()
                if not converged:
                    break

            if self.k.time >= self.k.t:
                break

        # the episode will be finished if it is not converged.
        done = not converged or (self.k.time == self.k.t)
        obs = self.get_state()
        infos = {}
        reward = self.compute_reward(rl_actions)

        return obs, reward, done, infos

    def get_state(self):
        return [0, 0, 0, 0, 0]

    def compute_reward(self, rl_actions, **kwargs):
        return 0


from pycigar.utils.input_parser import input_parser
misc_inputs_path = '../data/ieee37busdata/misc_inputs.csv'
dss_path = '../data/ieee37busdata/ieee37.dss'
load_solar_path = '../data/ieee37busdata/load_solar_data.csv'
breakpoints_path = '../data/ieee37busdata/breakpoints.csv'
sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, benchmark=False, percentage_hack=0.45, adv=False)

ini = []
reset = []
step = []
total = []

from pycigar.utils.logging import logger

for i in range(1):
    total_time = start_time = time.time()
    env = FooEnv(sim_params)
    ini.append(time.time() - start_time) #print("init environment: ", time.time() - start_time)
    start_time = time.time()
    env.reset()
    reset.append(time.time() - start_time) #print("reset environment: ", time.time() - start_time)
    done = False
    while not done:
        start_time = time.time()
        _, _, done, _ = env.step()
        step.append(time.time() - start_time) #print("step environment: ", time.time() - start_time)
    total.append(time.time() - total_time) #print("total environment: ", time.time() - total_time)

import numpy as np
print("init env: ", np.mean(np.array(ini)))
print("rest env: ", np.mean(np.array(reset)))
print("step env: ", np.mean(np.array(step)))
print("ttal env: ", np.mean(np.array(total)))

from pycigar.utils.logging import logger
print("odss env: ", logger().custom_metrics['opendss_time'])
print("scen env: ", logger().custom_metrics['update_scenario_time'])
print("node env: ", logger().custom_metrics['update_node_time'])
print("devi env: ", logger().custom_metrics['update_device_time'])
print("simu env: ", logger().custom_metrics['update_simulation_time'])
print("ctrl env: ", logger().custom_metrics['update_controller_time'])
"""

import multiprocessing
import time
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
from pycigar.utils.input_parser import input_parser
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from pycigar.utils.logging import logger
import os
import pycigar

start = 100
percentage_hack = 0.2

"""
Load the scenarios configuration file. This file contains the scenario information
for the experiment.
"""
misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, benchmark=True, percentage_hack=percentage_hack)
pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                    "env_name": "CentralControlPVInverterEnv",
                    "simulator": "opendss"}

create_env, env_name = make_create_env(pycigar_params, version=0)
register_env(env_name, create_env)
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
del sim_params['attack_randomization']

ini = []
reset = []
step = []
total = []

total_time = start_time = time.time()
test_env = create_env(sim_params)
test_env.observation_space  # get the observation space, we need this to construct our agent(s) observation input
test_env.action_space  # get the action space, we need this to construct our agent(s) action output
ini.append(time.time() - start_time) #print("init environment: ", time.time() - start_time)
done = False
start_time = time.time()
test_env.reset()
reset.append(time.time() - start_time) #print("reset environment: ", time.time() - start_time)
while not done:
    start_time = time.time()
    obs, r, done, _ = test_env.step(2)
    step.append(time.time() - start_time) #print("step environment: ", time.time() - start_time)
total.append(time.time() - total_time) #print("total environment: ", time.time() - total_time)

import numpy as np
print("init env: ", np.mean(np.array(ini)))
print("rest env: ", np.mean(np.array(reset)))
print("step env: ", np.mean(np.array(step)))
print("ttal env: ", np.mean(np.array(total)))

print("step only: ", logger().custom_metrics['step_only_time'])