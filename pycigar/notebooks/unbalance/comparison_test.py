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
from ray.rllib.models.tf.tf_action_dist import DiagGaussian
from ray.rllib.models.catalog import ModelCatalog
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def adjust_defaults(sim_params):
    for node in sim_params['scenario_config']['nodes']:
        for d in node['devices']:
            d['adversary_controller'] = 'unbalanced_fixed_controller'
            name = d['name']
            c = np.array(d['custom_device_configs']['default_control_setting'])
            if name.endswith('a'):
                c = c #- 0.02
            elif name.endswith('b'):
                c = c #+ 0.02
            elif name.endswith('c'):
                c = c #- 0.015
            d['custom_device_configs']['default_control_setting'] = c

    sim_params['simulation_config']['custom_configs']['solution_control_mode'] = 2


def set_hack_percent(sim_params, hack=0.45):
    for node in sim_params['scenario_config']['nodes']:
        for d in node['devices']:
            d['hack'] = [250, hack, 500]

def get_config(env_name='CentralControlPhaseSpecificContinuousPVInverterEnv'):
    pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                      'env_name': env_name,
                      'simulator': 'opendss'}
    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)

    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/breakpoints.csv"
    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)

    eval_start = 100
    sim_params['scenario_config']['start_end_time'] = [eval_start, eval_start + 750]
    sim_params['scenario_config']['multi_config'] = False
    del sim_params['attack_randomization']
    adjust_defaults(sim_params)

    return create_env, sim_params

def eval_dir_discrete(best_dir, eval_start=100, hack=0.45):
    create_env, sim_params = get_config('CentralControlPhaseSpecificPVInverterEnv')
    set_hack_percent(sim_params, hack)
    result_dict = {}
    sim_params['scenario_config']['start_end_time'] = [eval_start, eval_start + 750]

    test_env = create_env(sim_params)
    action_dist, _ = ModelCatalog.get_action_dist(
        test_env.action_space, config={}, dist_type=None, framework='tf')
    #assert isinstance(action_dist, DiagGaussian.__class__), 'For now only continuous gaussian action are supported'
    done = False
    obs = test_env.reset()
    obs = obs.tolist()
    while not done:
        out = infer(
            prev_reward=tf.constant([0.], tf.float32),
            observations=tf.constant([obs], tf.float32),
            is_training=tf.constant(False),
            seq_lens=tf.constant([0], tf.int32),
            prev_action=tf.constant([0], tf.int64)
        )['action_dist_inputs'].numpy()
        dist = action_dist(inputs=out, model=None)
        act = np.array(dist.deterministic_sample()).flatten()

        obs, r, done, _ = test_env.step(act)
        obs = obs.tolist()

    Logger = logger()
    f = plot_new(Logger.log_dict, Logger.custom_metrics, 0, unbalance=True)
    result_dict['figure'] = f
    plt.close(f)

    return result_dict



best_dir = Path('/home/toanngo/results_discrete_eval_30_random_reg_original_add_voltage_scale_2000_new_eval/main/run_train/run_train_0_P=100,M=50000,N=50,P=100_2020-08-22_21-43-376e6ljdl7/eval/1000/')

policy = tf.saved_model.load(str(best_dir / f'policy_1000'))
infer = policy.signatures['serving_default']

res = eval_dir_discrete(best_dir, 0, 0.45)
res['figure']