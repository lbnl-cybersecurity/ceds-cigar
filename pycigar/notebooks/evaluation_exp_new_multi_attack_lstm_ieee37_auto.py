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
import matplotlib
matplotlib.use('agg', force=True, warn=False)
import matplotlib.pyplot as plt
import json
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import NoopLogger, CSVLogger
from datetime import datetime
import tempfile
from uuid import uuid4

ActionTuple = namedtuple('Action', ['action', 'timestep'])


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs per trial')
    parser.add_argument('--save-path', type=str, default='~/hp_experiment3', help='where to save the results')
    parser.add_argument('--workers', type=int, default=7, help='number of cpu workers per run')
    parser.add_argument('--eval-rounds', type=int, default=1,
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
        f = plot_new(episodes[i].hist_data['logger']['log_dict'], episodes[i].hist_data['logger']['custom_metrics'], trainer.iteration, trainer.config['unbalance'])
        f.savefig(trainer.config['reporter_dir'] + 'eval-epoch-' + str(trainer.iteration) + '_' + str(i+1) + '.png',
                bbox_inches='tight')
        plt.close(f)
    save_best_policy(trainer, episodes)
    return metrics


# ==== CUSTOM METRICS (eval and training) ====
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

        # save plots
        ep = episodes[-1]
        data = ep.hist_data['logger']['log_dict']
        f = plot_new(data, ep.hist_data['logger']['custom_metrics'], trainer.iteration, trainer.config['unbalance'])
        f.savefig(os.path.join(trainer.config['reporter_dir'], 'best', 'eval.png'))
        plt.close(f)

    # save policy
    if True:
        shutil.rmtree(os.path.join(trainer.config['reporter_dir'], 'latest', 'policy'), ignore_errors=True)
        trainer.get_policy().export_model(os.path.join(trainer.config['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration)))
        trainer.save(os.path.join(trainer.config['reporter_dir'], 'checkpoint'))
        # save CSV
        #k = list(data.keys())[0]
        #ep_hist = pd.DataFrame(dict(v=data[data[k]['node']]['voltage'], y=data[k]['y'],
        #                            q_set=data[k]['q_set'], q_val=data[k]['q_out']))
        #a_hist = pd.DataFrame(data[k]['control_setting'], columns=['a1', 'a2', 'a3', 'a4', 'a5'])
        #adv_a_hist = pd.DataFrame(data['adversary_' + k]['control_setting'],
        #                          columns=['adv_a1', 'adv_a2', 'adv_a3', 'adv_a4', 'adv_a5'])
        #translation, slope = get_translation_and_slope(data[k]['control_setting'])
        #adv_translation, adv_slope = get_translation_and_slope(data['adversary_' + k]['control_setting'])
        #trans_slope_hist = pd.DataFrame(dict(translation=translation, slope=slope,
        #                                     adv_translation=adv_translation, adv_slope=adv_slope))

        #df = ep_hist.join(a_hist, how='outer')
        #df = df.join(adv_a_hist, how='outer')
        #df = df.join(trans_slope_hist, how='outer')
        #df.to_csv(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'last_eval_hists.csv'))

        # save info
        #start = ep.custom_metrics["hack_start"]
        #end = ep.custom_metrics["hack_end"]
        info = {
            'epoch': trainer.iteration,
        #    'hack_start': start,
        #    'hack_end': end,
            'reward': mean_r
        }
        with open(os.path.join(trainer.config['reporter_dir'], 'best', 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return CSVLogger(config, logdir)

    return logger_creator

def run_train(config, reporter):
    trainer_cls = APPOTrainer if config['algo'] == 'appo' else PPOTrainer
    trainer = trainer_cls(config=config['config'], logger_creator=custom_log_creator(os.path.expanduser(config['save_path'] + "/ray_results_{}_{}".format(datetime.today().strftime("%Y-%m-%d_%H-%M-%S"), uuid4())), 'custom_dir'))
    #trainer.restore('/home/toanngo/checkpoint_50/checkpoint-50')
    #trainer.restore('/home/toanngo/half_full_40/half_full_40_0/run_train/run_train_1_lr=0.001_2020-06-01_06-12-15zchm40ja/checkpoint/checkpoint_90/checkpoint-90')
    # needed so that the custom eval fn knows where to save plots
    trainer.config['reporter_dir'] = reporter.logdir
    trainer.config['unbalance'] = config['unbalance']

    for _ in tqdm(range(config['epochs'])):
        results = trainer.train()
        #if 'logger' in results['hist_stats']:
        #    del results['hist_stats']['logger']  # don't send to tensorboard
        #if 'evaluation' in results:
        #    del results['evaluation']['hist_stats']['logger']
        #reporter(**results)

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

def adjust_default_curves(base_config):
    for node in (
            base_config['env_config']['scenario_config']['nodes']
            + base_config['evaluation_config']['env_config']['scenario_config']['nodes']
    ):
        for d in node['devices']:
            name = d['name']
            c = np.array(d['custom_device_configs']['default_control_setting'])
            # found by training with no attack
            if name.endswith('a'):
                c = c + 0.04
            elif name.endswith('b'):
                c = c - 0.02
            elif name.endswith('c'):
                c = c + 0.01
            d['custom_device_configs']['default_control_setting'] = c
            d['custom_controller_configs']['default_control_setting'] = c
            d['adversary_custom_controller_configs']['default_control_setting'] = c
if __name__ == '__main__':
    args = parse_cli_args()

    if args.unbalance:
        pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                          'env_name': 'CentralControlPhaseSpecificPVInverterEnv',
                          'simulator': 'opendss'}
    else:
        pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                          'env_name': 'MultiAttackCentralControlPhaseSpecificPVInverterEnv',
                          'simulator': 'opendss'}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)

    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
    sim_params['vectorized_mode'] = True
    sim_params['is_disable_log'] = True
    sim_params_eval = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, benchmark=True)
    sim_params_eval['vectorized_mode'] = True
    sim_params_eval['is_disable_log'] = False
    base_config = {
        "env": env_name,
        "gamma": 0.5,
        'lr': 1e-3,
        #"lr_schedule": [[0, 2e-2], [20000, 1e-4]],
        'env_config': deepcopy(sim_params),
        'rollout_fragment_length': 40,
        'train_batch_size': 40*args.workers, #256, #250
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
            'use_lstm': True,
            'lstm_cell_size': 16,
            'max_seq_len': 5,
            'framestack': False,
            #'lstm_use_prev_reward': True,
            #'lstm_use_prev_action': True
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
    # eval environment should not be random across workers
    #base_config['env_config']['attack_randomization']['generator'] = 'UnbalancedAttackDefinitionGeneratorEvaluationRandom'
    #base_config['evaluation_config']['env_config']['attack_randomization']['generator'] = 'UnbalancedAttackDefinitionGeneratorEvaluation'
    base_config['env_config']['attack_randomization']['generator'] = 'AttackGenerator'
    base_config['evaluation_config']['env_config']['attack_randomization']['generator'] = 'AttackGeneratorEvaluation'
    #base_config['env_config']['scenario_config']['custom_configs']['load_scaling_factor'] = 1.5
    #base_config['evaluation_config']['env_config']['scenario_config']['custom_configs']['load_scaling_factor'] = 1.5
    #base_config['env_config']['env_config']['sims_per_step'] = 15
    #base_config['evaluation_config']['env_config']['sims_per_step'] = 15

    #base_config['env_config']['scenario_config']['custom_configs']['solar_scaling_factor'] = 3
    #base_config['evaluation_config']['env_config']['scenario_config']['custom_configs']['solar_scaling_factor'] = 3
    #base_config['env_config']['scenario_config']['custom_configs']['slack_bus_voltage'] = 1.04
    #base_config['evaluation_config']['env_config']['scenario_config']['custom_configs']['slack_bus_voltage'] = 1.04

    base_config['env_config']['M'] = 150
    base_config['env_config']['N'] = 0.5
    base_config['env_config']['P'] = 3
    base_config['env_config']['Q'] = 1
    base_config['env_config']['T'] = 20

    if args.unbalance:
        for node in base_config['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['adversary_controller'] = 'unbalanced_fixed_controller'
        for node in base_config['evaluation_config']['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['adversary_controller'] = 'unbalanced_fixed_controller'
    else:
        for node in base_config['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['adversary_controller'] = 'adaptive_auto_fixed_controller'
        for node in base_config['evaluation_config']['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['adversary_controller'] = 'adaptive_auto_fixed_controller'

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
        config['config']['lr'] = ray.tune.grid_search([5e-4])
        config['config']['env_config']['T'] = ray.tune.grid_search([0])
        #config['config']['model']['max_seq_len'] = ray.tune.grid_search([7])
        #config['config']['model']['lstm_use_prev_action_reward'] = ray.tune.grid_search([True])
        run_hp_experiment(config, 'main')

    ray.shutdown()
