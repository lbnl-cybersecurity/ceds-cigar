import argparse
import json
import math
import os
import pickle
import shutil
from collections import namedtuple
from copy import deepcopy

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.tune.registry import register_env
from tqdm import tqdm

import pycigar
from pycigar.utils.input_parser import input_parser
from pycigar.utils.logging import logger
from pycigar.utils.output import plot_new
from pycigar.utils.registry import make_create_env
import matplotlib.pyplot as plt

def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
    parser.add_argument('--save-path', type=str, default='~/hp_experiment3', help='where to save the results')
    return parser.parse_args()

def plot_all(log_dicts, custom_metrics, sim_params):
    colors = {(100, 0.2): 'tab:blue',
             (11000, 0.2): 'tab:purple',
             (100, 0.45): 'tab:orange',
             (11000, 0.45): 'tab:red',
            }

    rows = {(100, 0.2): 0,
             (11000, 0.2): 1,
             (100, 0.45): 2,
             (11000, 0.45): 3,
            }
    def get_translation_and_slope(a_val, init_a):
        points = np.array(a_val)
        slope = points[:, 1] - points[:, 0]
        translation = points[:, 2] - init_a[2]
        return translation, slope

    plt.rc('font', size=15)
    plt.rc('figure', titlesize=35)
    inv_k = 'inverter_s701a'
    f, ax = plt.subplots(11, 5, figsize=(60, 60))
    title = '[M {}][N {}][P {}][Q {}][sim {:d}]'.format(sim_params['M'], sim_params['N'], sim_params['P'], sim_params['Q'], sim_params['env_config']['sims_per_step'])
    f.suptitle(title)

    average_reward = {0: [],
                      1: [],
                      2: [],
                      3: [],
                      4: []
                      }

    for key in log_dicts.keys():
        ax[0, key[0]].plot(log_dicts[key][log_dicts[key][inv_k]['node']]['voltage'], color=colors[key[1:]], label='v ' + str(key[1:]))
        ax[1, key[0]].plot(log_dicts[key][inv_k]['y'], color=colors[key[1:]], label='y' + str(key[1:]))

        translation, slope = get_translation_and_slope(log_dicts[key][inv_k]['control_setting'], custom_metrics[key]['init_control_settings'][inv_k])
        ax[2, key[0]].plot(translation, color=colors[key[1:]], label='a '  + str(key[1:]))

        ax[3, key[0]].plot(log_dicts[key][inv_k]['sbar_solarirr'], color=colors[key[1:]], label='sbar solar '  + str(key[1:]))
        ax[4, key[0]].plot(log_dicts[key][inv_k]['sbar_pset'], color=colors[key[1:]], label='sbar pset ' + str(key[1:]))

        ax[5, key[0]].plot(log_dicts[key]['component_observation']['component_y'], color=colors[key[1:]], label='obs_y ' + str(key[1:]))
        ax[6, key[0]].plot(log_dicts[key]['component_observation']['component_ymax'], color=colors[key[1:]], label='obs_ymax ' + str(key[1:]))

        component_y = np.array(log_dicts[key]['component_reward']['component_y'])
        component_oa = np.array(log_dicts[key]['component_reward']['component_oa'])
        component_init = np.array(log_dicts[key]['component_reward']['component_init'])
        component_pset_pmax = np.array(log_dicts[key]['component_reward']['component_pset_pmax'])

        total_reward = (component_y + component_oa + component_init + component_pset_pmax)

        average_reward[key[0]].append(sum(total_reward)/sim_params['env_config']['sims_per_step'])
        ax[7 + rows[key[1:]], key[0]].plot(-component_y, label='abs_reward_y')
        ax[7 + rows[key[1:]], key[0]].plot(-component_oa, label='abs_reward_oa')
        ax[7 + rows[key[1:]], key[0]].plot(-component_init, label='abs_reward_init ')
        ax[7 + rows[key[1:]], key[0]].plot(-component_pset_pmax, label='abs_reward_pset_pmax')
        ax[7 + rows[key[1:]], key[0]].plot(-total_reward, label='abs_total_reward')
        ax[7 + rows[key[1:]], key[0]].set_title('{} total r: {:.2f}, r on s701a: {:.2f}'.format(key[1:], sum(total_reward)/sim_params['env_config']['sims_per_step'], sum(log_dicts[key][inv_k]['reward'])))

    for key in log_dicts.keys():
        avg = sum(average_reward[key[0]])/len(average_reward[key[0]])
        ax[0, key[0]].set_title('average eval: {:.2f}'.format(avg))

    for i in range(11):
        for j in range(5):
            ax[i,j].grid(b=True, which='both')
            ax[i,j].legend(loc=1, ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return f


def run_exp(config, reporter):
    sim_params = config['config']['env_config']
    test_env = create_env(sim_params)
    test_env.observation_space
    test_env.action_space

    log_dicts = {}
    custom_metrics = {}
    random_rl_update = None
    clone_random_rl_update = None

    for case in tqdm(range(5)):
        for i in tqdm(range(4)):
            done = False
            obs = test_env.reset()
            obs = obs.tolist()
            j = 0
            begin_hack = 13
            end_hack = 25
            act_step_down = 1
            act_step_up = 3
            act = act_init = 2
            while not done:
                j += 1
                if case == 0:
                    act = act_init

                if case == 1: # step down and no init
                    if j == begin_hack:
                        act = act_step_down

                elif case == 2: # step down up no init
                    if j == begin_hack and logger().custom_metrics['hack'] == 0.2:
                        act = act_step_up
                    if j == begin_hack and logger().custom_metrics['hack'] != 0.2:
                        act = act_step_down

                elif case == 3: # step down and init
                    if j == begin_hack:
                        act = act_step_down
                    if j == end_hack:
                        act = act_init

                elif case == 4: # step down up and init
                    if j == begin_hack and logger().custom_metrics['hack'] == 0.2:
                        act = act_step_up
                    if j == begin_hack and logger().custom_metrics['hack'] != 0.2:
                        act = act_step_down
                    if j == end_hack:
                        act = act_init

                if clone_random_rl_update:
                    obs, r, done, _ = test_env.step(act, clone_random_rl_update.pop(0))
                else:
                    obs, r, done, _ = test_env.step(act)
                obs = obs.tolist()
            log_dicts[(case, logger().custom_metrics['start_time'], logger().custom_metrics['hack'])] = deepcopy(logger().log_dict)
            custom_metrics[(case, logger().custom_metrics['start_time'], logger().custom_metrics['hack'])] = deepcopy(logger().custom_metrics)
            if not random_rl_update:
                random_rl_update = deepcopy(logger().custom_metrics['randomize_rl_update'])
            clone_random_rl_update = deepcopy(random_rl_update)
            print('Done case {} act {} M{} N{} P{} Q{}'.format(case, i, sim_params['M'], sim_params['N'], sim_params['P'], sim_params['Q']))

    f = plot_all(log_dicts, custom_metrics, sim_params)
    f.savefig(os.path.join(reporter.logdir, '..' ,'eval_M{}_N{}_P{}_Q{}.png'.format(sim_params['M'], sim_params['N'], sim_params['P'], sim_params['Q'])), facecolor='white')
    plt.close(f)

def run_hp_experiment(full_config):
    res = tune.run(run_exp,
                   config=full_config,
                   resources_per_trial={'cpu': 1
                                       },
                   local_dir=os.path.join(os.path.expanduser(full_config['save_path']))
                   )

if __name__ == '__main__':
    args = parse_cli_args()

    pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                        'env_name': 'CentralControlPVInverterEnv',
                        'simulator': 'opendss'}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)

    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)

    base_config = {
        'env_config': deepcopy(sim_params),
    }
    # eval environment should not be random across workers
    base_config['env_config']['attack_randomization']['generator'] = 'AttackDefinitionGeneratorEvaluation'

    ray.init(local_mode=False)

    full_config = {
        'config': base_config,
        'save_path': args.save_path,
    }

    config = deepcopy(full_config)
    #config['config']['env_config']['M'] = ray.tune.grid_search([10, 15, 20])
    #config['config']['env_config']['N'] = ray.tune.grid_search([0.1])
    #config['config']['env_config']['P'] = ray.tune.grid_search([18, 20, 25])
    #config['config']['env_config']['Q'] = ray.tune.grid_search([100, 250, 500, 1000])
    config['config']['env_config']['M'] = ray.tune.grid_search([10])
    config['config']['env_config']['N'] = ray.tune.grid_search([0.1])
    config['config']['env_config']['P'] = ray.tune.grid_search([18])
    config['config']['env_config']['Q'] = ray.tune.grid_search([100])
    run_hp_experiment(config)

    ray.shutdown()