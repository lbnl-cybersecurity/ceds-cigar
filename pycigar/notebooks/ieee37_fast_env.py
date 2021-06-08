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
import matplotlib
matplotlib.use('agg', force=True, warn=False)
import matplotlib.pyplot as plt
from ray.rllib.agents.callbacks import DefaultCallbacks

ActionTuple = namedtuple('Action', ['action', 'timestep'])


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs per trial')
    parser.add_argument('--save-path', type=str, default='~/hp_experiment3', help='where to save the results')
    parser.add_argument('--workers', type=int, default=1, help='number of cpu workers per run')
    parser.add_argument('--eval-rounds', type=int, default=1,
                        help='number of evaluation rounds to run to smooth random results')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='do an evaluation every N epochs')
    parser.add_argument("--algo", help="use PPO or APPO", choices=['ppo', 'appo'],
                        nargs='?', const='ppo', default='ppo', type=str.lower)
    parser.add_argument('--unbalance', dest='unbalance', action='store_true')

    return parser.parse_args()


def plot(log_dict, custom_metrics, iteration):
    target = '701'
    f, ax = plt.subplots(4, 1, figsize=(25/3, 4*4))
    plt.rc('font', size=15)
    plt.rc('figure', titlesize=15)
    f.suptitle('[hack {}][time {}][epoch {}] Bus {} IEEE-37 feeder'.format(custom_metrics['hack_percentage'], custom_metrics['start_time'], iteration, target))
    ax[0].plot(log_dict['v_metrics'][target])
    ax[0].set_ylabel('p.u. Voltage')
    #ax[0].set_ylim(1.00, 1.06)

    ax[1].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='u_worst')
    ax[1].plot(np.array(log_dict['u_metrics']['u_mean'])*100, label='u_mean')
    ax[1].plot(np.array(log_dict['u_metrics'][target])*100, label=target)
    ax[1].set_ylabel('imbalance percent')
    ax[1].set_ylim(0, 8)

    ax[2].plot(log_dict['a_metrics']['a'], label='a')
    ax[2].plot(log_dict['a_metrics']['b'], label='b')
    ax[2].plot(log_dict['a_metrics']['c'], label='c')
    ax[2].set_ylim(-0.1, 0.1)

    for reg in log_dict['reg_metrics']:
        ax[3].plot(log_dict['reg_metrics'][reg])
    ax[3].set_ylim(-10, 17)

    for a in ax:
        a.grid(b=True, which='both')
        a.legend(loc=1, ncol=1)


    ax[3].set_xlabel('time (s)')
    #plt.yticks(np.arange(0.90, 1.04, step=0.04))
    ax[0].legend(['a', 'b', 'c'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    return f

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
        f = plot(episodes[i].hist_data['logger']['log_dict'], episodes[i].hist_data['logger']['custom_metrics'], trainer.iteration)
        f.savefig(trainer.config['reporter_dir'] + 'eval-epoch-' + str(trainer.iteration) + '_' + str(i+1) + '.png',
                bbox_inches='tight')
        plt.close(f)
    save_best_policy(trainer, episodes)
    return metrics


class CustomCallbacks(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict=None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        self.ActionTuple = namedtuple('Action', ['action', 'timestep'])

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        pass

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        pass

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        tracking = logger()
        episode.hist_data['logger'] = {'log_dict': tracking.log_dict, 'custom_metrics': tracking.custom_metrics}


def get_translation_and_slope(a_val):
    points = np.array(a_val)
    slope = points[:, 1] - points[:, 0]
    og_point = points[0, 2]
    translation = points[:, 2] - og_point
    return translation, slope


def save_best_policy(trainer, episodes):
    mean_r = np.array([ep.episode_reward for ep in episodes]).mean()
    if 'best_eval_reward' not in trainer.config or trainer.config['best_eval_reward'] < mean_r:
        os.makedirs(os.path.join(trainer.config['reporter_dir'], 'best'), exist_ok=True)
        trainer.config['best_eval_reward'] = mean_r
        # save policy
        if True:
            shutil.rmtree(os.path.join(trainer.config['reporter_dir'], 'best', 'policy'), ignore_errors=True)
            trainer.get_policy().export_model(os.path.join(trainer.config['reporter_dir'], 'best', 'policy' + '_' + str(trainer.iteration)))

    # save policy
    if True:
        shutil.rmtree(os.path.join(trainer.config['reporter_dir'], 'latest', 'policy'), ignore_errors=True)
        trainer.get_policy().export_model(os.path.join(trainer.config['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration)))
        trainer.save(os.path.join(trainer.config['reporter_dir'], 'checkpoint'))
        info = {
            'epoch': trainer.iteration,
            'reward': mean_r
        }
        with open(os.path.join(trainer.config['reporter_dir'], 'best', 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)


def run_train(config, reporter):
    trainer_cls = APPOTrainer if config['algo'] == 'appo' else PPOTrainer
    trainer = trainer_cls(config=config['config'])
    trainer.config['reporter_dir'] = reporter.logdir
    #trainer.restore('/global/scratch/sytoanngo/checkpoint_u_worst/checkpoint-1300')

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
                      'env_name': 'SimpleEnv',
                      'simulator': 'opendss'}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)

    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/breakpoints.csv"

    sim_params = {'misc_inputs_path': misc_inputs_path,
                'dss_path': dss_path,
                'load_solar_path': load_solar_path,
                'breakpoints_path': breakpoints_path,
                'sim_per_step': 30,
                'mode': 'training',
                'log': False,
                'M': 100,
                'N': 10,
                'P': 1,
                'Q': 10}
    sim_params_eval = {'misc_inputs_path': misc_inputs_path,
                'dss_path': dss_path,
                'load_solar_path': load_solar_path,
                'breakpoints_path': breakpoints_path,
                'sim_per_step': 30,
                'mode': 'eval',
                'log': True,
                'M': 100,
                'N': 10,
                'P': 1,
                'Q': 10}

    base_config = {
        "env": env_name,
        "gamma": 0.5,
        'lr': 5e-4,
        #"lr_schedule": [[0, 2e-2], [20000, 1e-4]],
        'env_config': deepcopy(sim_params),
        'rollout_fragment_length': 20,
        'sgd_minibatch_size': 20,
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
            'use_lstm': False,
            'framestack': False,
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
        "callbacks": CustomCallbacks
    }

    base_config['env_config']['M'] = 150
    base_config['env_config']['N'] = 1 #roa
    base_config['env_config']['P'] = 2 #init
    base_config['env_config']['Q'] = 1

    ray.init(local_mode=False)

    full_config = {
        'config': base_config,
        'epochs': args.epochs,
        'save_path': args.save_path,
        'algo': args.algo,
        'unbalance': args.unbalance
    }

    for i in range(1):
        config = deepcopy(full_config)
        #config['config']['env_config']['M'] = ray.tune.grid_search([300, 350, 400, 500])
        #config['config']['env_config']['N'] = ray.tune.grid_search([1])
        #config['config']['env_config']['P'] = ray.tune.grid_search([1])
        #config['config']['env_config']['Q'] = ray.tune.grid_search([1])
        #config['config']['lr'] = ray.tune.grid_search([1e-4, 5e-4]) 

        run_hp_experiment(config, 'main')

    ray.shutdown()