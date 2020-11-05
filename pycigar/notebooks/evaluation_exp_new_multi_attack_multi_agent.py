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
from pycigar.utils.output import plot_new, plot_cluster
from pycigar.utils.registry import make_create_env
import matplotlib.pyplot as plt

ActionTuple = namedtuple('Action', ['action', 'timestep'])


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs per trial')
    parser.add_argument('--save-path', type=str, default='~/hp_experiment3', help='where to save the results')
    parser.add_argument('--workers', type=int, default=7, help='number of cpu workers per run')
    parser.add_argument('--eval-rounds', type=int, default=4,
                        help='number of evaluation rounds to run to smooth random results')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='do an evaluation every N epochs')
    parser.add_argument("--algo", help="use PPO or APPO", choices=['ppo', 'appo'],
                        nargs='?', const='ppo', default='ppo', type=str.lower)
    parser.add_argument('--unbalance', dest='unbalance', action='store_true')

    return parser.parse_args()


def custom_eval_function(trainer, eval_workers):
    if trainer.config["evaluation_num_workers"] == 0:
        for _ in range(trainer.config["evaluation_num_episodes"]):
            eval_workers.local_worker().sample()

    else:
        num_rounds = int(math.ceil(trainer.config["evaluation_num_episodes"] /
                                   trainer.config["evaluation_num_workers"]))
        for i in range(num_rounds):
            ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    episodes, _ = collect_episodes(eval_workers.local_worker(), eval_workers.remote_workers())
    metrics = summarize_episodes(episodes)

    for i in range(len(episodes)):
        f = plot_cluster(episodes[i].hist_data['logger']['log_dict'], episodes[i].hist_data['logger']['custom_metrics'], trainer.iteration, trainer.global_vars['unbalance'])
        f.savefig(trainer.global_vars['reporter_dir'] + 'eval-epoch-' + str(trainer.iteration) + '_' + str(i+1) + '.png',
                bbox_inches='tight')
        plt.close(f)
    save_best_policy(trainer, episodes)
    return metrics


# ==== CUSTOM METRICS (eval and training) ====
def on_episode_start(info):
    pass


def on_episode_step(info):
    pass


def on_episode_end(info):
    episode = info["episode"]
    tracking = logger()
    info['episode'].hist_data['logger'] = {'log_dict': tracking.log_dict, 'custom_metrics': tracking.custom_metrics}


def get_translation_and_slope(a_val):
    points = np.array(a_val)
    slope = points[:, 1] - points[:, 0]
    og_point = points[0, 2]
    translation = points[:, 2] - og_point
    return translation, slope


def save_best_policy(trainer, episodes):
    mean_r = np.array([ep.episode_reward for ep in episodes]).mean()
    if 'best_eval_reward' not in trainer.global_vars or trainer.global_vars['best_eval_reward'] < mean_r:
        os.makedirs(os.path.join(trainer.global_vars['reporter_dir'], 'best'), exist_ok=True)
        trainer.global_vars['best_eval_reward'] = mean_r
        # save policy
        if not trainer.global_vars['unbalance']:
            shutil.rmtree(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'policy'), ignore_errors=True)
            trainer.get_policy('agent_1').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'policy' + '_' + str(trainer.iteration), 'agent_1'))
            trainer.get_policy('agent_2').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'policy' + '_' + str(trainer.iteration), 'agent_2'))
            trainer.get_policy('agent_3').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'policy' + '_' + str(trainer.iteration), 'agent_3'))

        # save plots
        ep = episodes[-1]
        data = ep.hist_data['logger']['log_dict']
        f = plot_cluster(data, ep.hist_data['logger']['custom_metrics'], trainer.iteration, trainer.global_vars['unbalance'])
        f.savefig(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'eval.png'))
        plt.close(f)

    # save policy
    if not trainer.global_vars['unbalance']:
        shutil.rmtree(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy'), ignore_errors=True)
        trainer.get_policy('agent_1').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration), 'agent_1'))
        trainer.get_policy('agent_2').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration), 'agent_2'))
        trainer.get_policy('agent_3').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration), 'agent_3'))


def run_train(config, reporter):
    trainer_cls = APPOTrainer if config['algo'] == 'appo' else PPOTrainer
    trainer = trainer_cls(config=config['config'])
    #trainer.restore('/home/toanngo/checkpoint_50/checkpoint-50')
    #trainer.restore('/home/toanngo/half_full_40/half_full_40_0/run_train/run_train_1_lr=0.001_2020-06-01_06-12-15zchm40ja/checkpoint/checkpoint_90/checkpoint-90')
    # needed so that the custom eval fn knows where to save plots
    trainer.global_vars['reporter_dir'] = reporter.logdir
    trainer.global_vars['unbalance'] = config['unbalance']

    for _ in tqdm(range(config['epochs'])):
        results = trainer.train()
        if 'logger' in results['hist_stats']:
            del results['hist_stats']['logger']  # don't send to tensorboard
        if 'evaluation' in results:
            del results['evaluation']['hist_stats']['logger']
        reporter(**results)

    trainer.stop()


def run_hp_experiment(full_config, name):
    res = tune.run(run_train,
                   config=full_config,
                   resources_per_trial={'cpu': 1, 'gpu': 0,
                                        'extra_cpu': full_config['config']['num_workers']
                                                     + full_config['config']['evaluation_num_workers']},
                   local_dir=os.path.join(os.path.expanduser(full_config['save_path']), name)
                   )
    # save results
    with open(os.path.join(os.path.expanduser(full_config['save_path']), str(name) + '.pickle'), 'wb') as f:
        pickle.dump(res.trial_dataframes, f)


if __name__ == '__main__':
    args = parse_cli_args()

    pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                        'env_name': 'ClusterMultiEnv',
                        'simulator': 'opendss'}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)


    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
    sim_params['vectorized_mode'] = True
    sim_params['is_disable_log'] = True
    sim_params['cluster'] = {'1': ['s701a', 's701b', 's701c', 's712c', 's713c', 's714a', 's714b', 's718a', 's720c', 's722b', 's722c', 's724b', 's725b'],
                             '2': ['s727c', 's728', 's729a', 's730c', 's731b', 's732c', 's733a'],
                             '3': ['s734c', 's735c', 's736b', 's737a', 's738a', 's740c', 's741c', 's742a', 's742b', 's744a']
                            }
    sim_params_eval = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, benchmark=True)
    sim_params_eval['vectorized_mode'] = True
    sim_params_eval['is_disable_log'] = False
    sim_params['cluster'] = {'1': ['s701a', 's701b', 's701c', 's712c', 's713c', 's714a', 's714b', 's718a', 's720c', 's722b', 's722c', 's724b', 's725b'],
                             '2': ['s727c', 's728', 's729a', 's730c', 's731b', 's732c', 's733a'],
                             '3': ['s734c', 's735c', 's736b', 's737a', 's738a', 's740c', 's741c', 's742a', 's742b', 's744a']
                            }
    test_env = create_env(sim_params)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def policy_mapping_fn(agent_id):
        return 'agent_' + str(agent_id)

    base_config = {
        "env": env_name,
        "gamma": 0.5,
        'lr': 1e-3,
        #"lr_schedule": [[0, 2e-2], [20000, 1e-4]],
        'env_config': deepcopy(sim_params),
        'rollout_fragment_length': 20,
        'train_batch_size': 20*args.workers, #256, #250
        'clip_param': 0.1,
        'lambda': 0.95,
        'vf_clip_param': 100,

        'num_workers': args.workers,
        'num_cpus_per_worker': 1,
        'num_cpus_for_driver': 1,
        'num_envs_per_worker': 1,

        'log_level': 'WARNING',

        'model': {
            'fcnet_activation': 'tanh',
            'fcnet_hiddens': [32, 32], #[16, 16],
            'free_log_std': False,
            'vf_share_layers': True,
            'use_lstm': False
        },

        'multiagent': {
            'policies': {'agent_1': (None, obs_space, act_space, {}),
                         'agent_2': (None, obs_space, act_space, {}),
                         'agent_3': (None, obs_space, act_space, {}),},
            'policy_mapping_fn': policy_mapping_fn,
        },
        # ==== EXPLORATION ====
        'explore': True,
        'exploration_config': {
            'type': 'StochasticSampling',  # default for PPO
        },

        # ==== EVALUATION ====
        "evaluation_num_workers": 1,
        'evaluation_num_episodes': args.eval_rounds,
        "evaluation_interval": args.eval_interval,
        "custom_eval_function": custom_eval_function,
        'evaluation_config': {
            "seed": 42,
            # IMPORTANT NOTE: For policy gradients, this might not be the optimal policy
            'explore': False,
            'env_config': deepcopy(sim_params_eval),
        },

        # ==== CUSTOM METRICS ====
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
        },
    }
    # eval environment should not be random across workers
    base_config['env_config']['attack_randomization']['generator'] = 'AttackDefinitionGeneratorEvaluationRandom'
    base_config['evaluation_config']['env_config']['attack_randomization']['generator'] = 'AttackDefinitionGeneratorEvaluation'

    #base_config['env_config']['scenario_config']['custom_configs']['load_scaling_factor'] = 1.5
    #base_config['evaluation_config']['env_config']['scenario_config']['custom_configs']['load_scaling_factor'] = 1.5

    #base_config['env_config']['scenario_config']['custom_configs']['solar_scaling_factor'] = 3
    #base_config['evaluation_config']['env_config']['scenario_config']['custom_configs']['solar_scaling_factor'] = 3
    #base_config['env_config']['scenario_config']['custom_configs']['slack_bus_voltage'] = 1.04
    #base_config['evaluation_config']['env_config']['scenario_config']['custom_configs']['slack_bus_voltage'] = 1.04

    base_config['env_config']['M'] = 25
    base_config['env_config']['N'] = 0.1
    base_config['env_config']['P'] = 9
    base_config['env_config']['Q'] = 25
    base_config['env_config']['T'] = 100

    ray.init(local_mode=False)

    full_config = {
        'config': base_config,
        'epochs': args.epochs,
        'save_path': args.save_path,
        'algo': args.algo,
        'unbalance': args.unbalance
    }

    for i in range(2):
        config = deepcopy(full_config)
        config['config']['lr'] = ray.tune.grid_search([1e-3])
        run_hp_experiment(config, 'main')

    ray.shutdown()
