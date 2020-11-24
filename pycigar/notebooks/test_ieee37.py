"""from pycigar.utils.input_parser import input_parser
from pycigar.notebooks.custom_pv_device import CustomPVDevice
from pycigar.utils.registry import register_devcon
misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'

dss = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
dss_b = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37_b.dss'
dss_c = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37_c.dss'

load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'


register_devcon('custom_pv_device', CustomPVDevice)

start = 100
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=False, percentage_hack=0.45)
sim_params['simulation_config']['network_model_directory'] = [dss, dss_b, dss_c]
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
sim_params['simulation_config']['custom_configs']['solution_control_mode'] = 2

del sim_params['attack_randomization']

from pycigar.envs import CentralControlPVInverterEnv
from pycigar.envs import CentralControlPhaseSpecificPVInverterEnv

#env = CentralControlPVInverterEnv(sim_params=sim_params)
env = CentralControlPhaseSpecificPVInverterEnv(sim_params=sim_params)

env.reset()
done = False
while not done:
    _, _, done, _ = env.step([5, 5, 5])

env.reset()

done = False
while not done:
    _, _, done, _ = env.step([5, 5, 5])

env.reset()

done = False
while not done:
    _, _, done, _ = env.step([5, 5, 5])"""

from gym.spaces import Tuple, Discrete, Box
from pycigar.utils.input_parser import input_parser
import pycigar
from pycigar.utils.registry import make_create_env
from pycigar.utils.input_parser import input_parser
from pycigar.utils.logging import logger
from pycigar.utils.output import plot_new

from ray.tune.registry import register_env

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
import json
from pathlib import Path
from ray.rllib.models.catalog import ModelCatalog
from tqdm import tqdm

import matplotlib.pyplot as plt
from pathlib import Path

from pycigar.utils.input_parser import input_parser
misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
dss = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'
#best_dir = Path('/home/toanngo/final_oscillation_results_discrete_eval_35_random_reg_no_max/final_results_discrete_eval_30_random_reg_no_max/main/run_train/run_train_0_T=2,M=50000,N=50,P=60,lr=0.0001_2020-09-16_22-45-03nu6061l6/best/reward_sum/policy_800')
#best_dir = Path('/home/toanngo/final_results_discrete_eval_30_current_u_30/final_results_discrete_eval_30_random_reg_no_max_30/main/run_train/run_train_0_T=2,M=50000,N=50,P=60,lr=0.0001_2020-09-18_10-46-38gbmacysj/eval/850/policy_850') # current u, step=30
#best_dir = Path('/home/toanngo/final_64_64_32_larger_last_action/final_results_discrete_eval_30_random_reg_no_max_64_64_32/main/run_train/run_train_1_T=2,M=50000,N=50,P=65,lr=0.0001_2020-09-19_00-22-44bwog3dhn/eval/1000/policy_1000') #current u, bigger last action penalty

#best_dir = Path('/home/toanngo/final_u_mean_p_65/final_results_discrete_eval_30_random_reg_no_max_64_64_32_new_u_p_65/main/run_train/run_train_2_T=2,M=50000,N=50,P=65,lr=0.0001_2020-09-21_17-50-12nxlnwtqb/best/reward_sum/policy_650') #u_mean reward 
#best_dir = Path('/home/toanngo/final_u_mean_p_70/final_results_discrete_eval_30_random_reg_no_max_64_64_32_new_u_p_70/main/run_train/run_train_0_T=2,M=50000,N=50,P=75,lr=0.0001_2020-09-22_08-51-0904bft2r_/best/reward_sum/policy_750')

best_dir = '/home/toanngo/Documents/GitHub/ceds-cigar-external/pycigar/docs/SAMPLE_RESULT_IEEE37_oscillation_policy'

start = 100
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=False, percentage_hack=0.3)
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
sim_params['env_config']['sim_per_step'] = 30
del sim_params['attack_randomization']


"""for node in sim_params['scenario_config']['nodes']:
    for d in node['devices']:
        d['adversary_controller'] = 'unbalanced_fixed_controller'"""

policy = tf.saved_model.load(str(best_dir))
infer = policy.signatures['serving_default']
action_dist, _ = ModelCatalog.get_action_dist(
    Tuple([Discrete(21)] * 3), config={}, dist_type=None, framework='tf')

from pycigar.envs.multiagent.multi_env_distributed_unb import UnbMultiEnv

env = UnbMultiEnv(sim_params=sim_params)
obs = env.reset()
done = False
init_action = {k: env.INIT_ACTION[k] for k in env.k.device.get_rl_device_ids()}
init_action_rl = {k: [10, 10, 10] for k in env.k.device.get_rl_device_ids()}
def encode_action(act):
    encode_act = {}
    for k in act:
        old_a_encoded = np.zeros(21*3)
        offsets = np.cumsum([0, *[a.n for a in Tuple([Discrete(21)] * 3)][:-1]])
        for action, offset in zip(act[k], offsets):
            old_a_encoded[offset + action] = 1
        encode_act[k] = old_a_encoded
    return encode_act

def decode_action(rl_act):
    decode_act = {}
    for k in rl_act:
        if k.endswith('a'):
            translation = rl_act[k][0]
        elif k.endswith('b'):
            translation = rl_act[k][1]
        elif k.endswith('c'):
            translation = rl_act[k][2]
        else:
            translation = int(21 / 2)

        decode_act[k]  = init_action[k] - 0.1 + 0.01 * translation
    return decode_act

last_act = init_action_rl
while not done:
    new_act = {}
    encode_act = encode_action(last_act)
    for k in env.k.device.get_rl_device_ids():
        obs_array = np.array([obs[k]['y_max'], obs[k]['y'], obs[k]['u'], obs[k]['sbar_solar_irr'], *encode_act[k], *obs[k]['voltage']]).tolist()
        out = infer(
            prev_reward=tf.constant([0.], tf.float32),
            observations=tf.constant([obs_array], tf.float32),
            is_training=tf.constant(False),
            seq_lens=tf.constant([0], tf.int32),
            prev_action=tf.constant([0], tf.int64)
        )['action_dist_inputs'].numpy()
        dist = action_dist(inputs=out, model=None)
        #act = dist.deterministic_sample().numpy().batches
        act = np.array(dist.deterministic_sample()).flatten()
        new_act[k] = act

    act = decode_action(new_act)
    last_act = new_act
    obs, r, done, _ = env.step(act)
    done = done['__all__']