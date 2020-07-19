import argparse

import yaml

SAVE_RATE = 5

"""
Parser to pass argument from terminal command
--run: RL algorithm, ex. PG, PPO, IMPALA
--stop: stop criteria of experiment. The experiment will stop when mean reward reach to this value.
Example of terminal command:
  > python single_relative_discrete_2_lr.py --run PPO --stop 0
"""
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=0)

"""
Load the scenarios configuration file. This file contains the scenario information
for the experiment.
"""
stream = open("../reg_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)
regs = sim_params['scenario_config']['regulators']
print(regs[0]['devices'])
# print((regs['devices'][0]['custom_configs']['default_control_setting']))

"""
Register the environment to OpenGymAI. This is necessary, RLlib can find the new environment
with string name env_name_v:version:, ex. SingleRelativeDiscreteCoopEnv_v0.
env_name: name of environment being used.
sim_params: simulation params, it is the scenario configuration.
simulator: the simulator being used, ex. opendss, gridlabd...
tracking_ids: list of ids of devices being tracked during the experiment.
"""
