# TEST FULL
from pycigar.utils.input_parser import input_parser
import numpy as np
from pycigar.utils.registry import register_devcon
import tensorflow as tf
from ray.rllib.models.catalog import ModelCatalog
from gym.spaces import Tuple, Discrete, Box
from copy import deepcopy
from pycigar.envs import SimpleEnv

misc_inputs_path = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
dss_path = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
load_solar_path = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
breakpoints_path = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'

sim_params = {'misc_inputs_path': misc_inputs_path,
              'dss_path': dss_path,
              'load_solar_path': load_solar_path,
              'breakpoints_path': breakpoints_path,
              'sim_per_step': 30,
              'mode': 'fix',
              'hack_percentage': 0.2,
              'start_time': 100,
              'log': True,
              'M': 100,
              'N': 10,
              'P': 1,
              'Q': 10}

env = SimpleEnv(sim_params)
obs = env.reset()
act_all = []
act = {}
done = False
t = 0
while not done:
    obs, reward, done, info = env.step([10, 10, 10])
