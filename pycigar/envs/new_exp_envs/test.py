from pycigar.utils.input_parser import input_parser
import numpy as np
from pycigar.utils.registry import register_devcon
import tensorflow as tf
from ray.rllib.models.catalog import ModelCatalog
from gym.spaces import Tuple, Discrete, Box

misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/misc_inputs.csv'
dss = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/ieee123.dss'
load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/load_solar_data.csv'
breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/breakpoints.csv'

start = 100
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=0.4)
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
del sim_params['attack_randomization']
for node in sim_params['scenario_config']['nodes']:
    node['devices'][0]['adversary_controller'] =  'unbalanced_fixed_controller'

from pycigar.envs.new_exp_envs.envs import ExpCentralControlPhaseSpecificPVInverterEnv
policy = "/home/toanngo/oscillation_ieee123_unbalance/main/run_train/run_train_1_T=150,lr=0.0004_2020-11-20_03-16-262pefpizn/best/policy_600"
env = ExpCentralControlPhaseSpecificPVInverterEnv(sim_params=sim_params)
env.reset()
done = False

for _ in range(150):
    _, r, done, _ = env.step([10, 10, 10])

print('ahihi')
for _ in range(15):
    _, r, done, _ = env.step([5, 10, 10])


for _ in range(100):
    _, r, done, _ = env.step([10, 10, 10])

while not done:
    _, r, done, _ = env.step([10, 10, 10])