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
from coma import CCTrainer
from coma_model import CentralizedCriticModel
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.tune.registry import register_env
from tqdm import tqdm
from ray.rllib.models import ModelCatalog
import pycigar
from pycigar.utils.input_parser import input_parser
from pycigar.utils.logging import logger
from pycigar.utils.output import plot_new, plot_cluster, save_cluster_csv, save_cluster_csv_lite
from pycigar.utils.registry import make_create_env
import matplotlib.pyplot as plt
import json

ActionTuple = namedtuple('Action', ['action', 'timestep'])


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs per trial')
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
        #f = plot_cluster(episodes[i].hist_data['logger']['log_dict'], episodes[i].hist_data['logger']['custom_metrics'], trainer.iteration, trainer.global_vars['unbalance'])
        #f.savefig(trainer.global_vars['reporter_dir'] + 'eval-epoch-' + str(trainer.iteration) + '_' + str(i+1) + '.png',
        #        bbox_inches='tight')
        #plt.close(f)
        result = save_cluster_csv(episodes[i].hist_data['logger']['log_dict'], episodes[i].hist_data['logger']['custom_metrics'], trainer.iteration, trainer.global_vars['unbalance'])
        with open(trainer.global_vars['reporter_dir'] + 'eval-epoch-' + str(trainer.iteration) + '_' + str(i+1) + '.json', 'w') as f:  # You will need 'wb' mode in Python 2.x
            json.dump(result, f)
        result = save_cluster_csv_lite(episodes[i].hist_data['logger']['log_dict'], episodes[i].hist_data['logger']['custom_metrics'], trainer.iteration, trainer.global_vars['unbalance'])
        with open(trainer.global_vars['reporter_dir'] + 'lite_eval-epoch-' + str(trainer.iteration) + '_' + str(i+1) + '.json', 'w') as f:  # You will need 'wb' mode in Python 2.x
            json.dump(result, f)
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
            trainer.get_policy('agent_4').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'policy' + '_' + str(trainer.iteration), 'agent_4'))
            trainer.get_policy('agent_5').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'policy' + '_' + str(trainer.iteration), 'agent_5'))

    # save policy
    if not trainer.global_vars['unbalance']:
        shutil.rmtree(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy'), ignore_errors=True)
        trainer.get_policy('agent_1').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration), 'agent_1'))
        trainer.get_policy('agent_2').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration), 'agent_2'))
        trainer.get_policy('agent_3').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration), 'agent_3'))
        trainer.get_policy('agent_4').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration), 'agent_4'))
        trainer.get_policy('agent_5').export_model(os.path.join(trainer.global_vars['reporter_dir'], 'latest', 'policy' + '_' + str(trainer.iteration), 'agent_5'))

    trainer.save(os.path.join(trainer.global_vars['reporter_dir'], 'checkpoint'))

def run_train(config, reporter):
    trainer_cls = CCTrainer
    trainer = trainer_cls(config=config['config'])
    trainer.global_vars['reporter_dir'] = reporter.logdir
    trainer.global_vars['unbalance'] = config['unbalance']
    #trainer.restore('/home/toanngo/ieee123_multi_agent_multi_attack_vf_coeff/main/run_train/run_train_90d65_00000_0_lr=0.0001_2021-02-26_01-39-48/checkpoint/checkpoint_1000/checkpoint-1000')

    for _ in tqdm(range(config['epochs'])):
        results = trainer.train()
        if 'logger' in results['hist_stats']:
            del results['hist_stats']['logger']  # don't send to tensorboard
        if 'evaluation' in results and 'logger' in results['evaluation']['hist_stats']:
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
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                        'env_name': 'ClusterMultiEnv',
                        'simulator': 'opendss'}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)


    misc_inputs_path = pycigar.DATA_DIR + "/ieee123busdata/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee123busdata/ieee123.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee123busdata/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee123busdata/breakpoints.csv"


    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
    sim_params['vectorized_mode'] = True
    sim_params['is_disable_log'] = True
    sim_params['cluster'] = {'1': ['s1a', 's2b', 's4c', 's5c', 's6c', 's7a', 's9a', 's10a', 's11a', 's12b', 's16c', 's17c', 's34c'],
                             '2': ['s52a', 's53a', 's55a', 's56b', 's58b', 's59b', 's86b', 's87b', 's88a', 's90b', 's92c', 's94a', 's95b', 's96b'],
                             '3': ['s19a', 's20a', 's22b', 's24c', 's28a', 's29a', 's30c', 's31c', 's32c', 's33a', 's35a', 's37a', 's38b', 's39b', 's41c', 's42a', 's43b', 's45a', 's46a', 's47', 's48', 's49a', 's49b', 's49c', 's50c', 's51a'],
                             '4': ['s102c', 's103c', 's104c', 's106b', 's107b', 's109a', 's111a', 's112a', 's113a', 's114a'],
                             '5': ['s60a', 's62c', 's63a', 's64b', 's65a', 's65b', 's65c', 's66c', 's68a', 's69a', 's70a', 's71a', 's73c', 's74c', 's75c', 's76a', 's76b', 's76c', 's77b', 's79a', 's80b', 's82a', 's83c', 's84c', 's85c', 's98a', 's99b', 's100c']}
    sim_params_eval = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, benchmark=True)
    sim_params_eval['vectorized_mode'] = True
    sim_params_eval['is_disable_log'] = False
    sim_params_eval['cluster'] = {'1': ['s1a', 's2b', 's4c', 's5c', 's6c', 's7a', 's9a', 's10a', 's11a', 's12b', 's16c', 's17c', 's34c'],
                             '2': ['s52a', 's53a', 's55a', 's56b', 's58b', 's59b', 's86b', 's87b', 's88a', 's90b', 's92c', 's94a', 's95b', 's96b'],
                             '3': ['s19a', 's20a', 's22b', 's24c', 's28a', 's29a', 's30c', 's31c', 's32c', 's33a', 's35a', 's37a', 's38b', 's39b', 's41c', 's42a', 's43b', 's45a', 's46a', 's47', 's48', 's49a', 's49b', 's49c', 's50c', 's51a'],
                             '4': ['s102c', 's103c', 's104c', 's106b', 's107b', 's109a', 's111a', 's112a', 's113a', 's114a'],
                             '5': ['s60a', 's62c', 's63a', 's64b', 's65a', 's65b', 's65c', 's66c', 's68a', 's69a', 's70a', 's71a', 's73c', 's74c', 's75c', 's76a', 's76b', 's76c', 's77b', 's79a', 's80b', 's82a', 's83c', 's84c', 's85c', 's98a', 's99b', 's100c']}
    test_env = create_env(sim_params)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def policy_mapping_fn(agent_id):
        return 'agent_' + str(agent_id)

    base_config = {
        "env": env_name,
        "batch_mode": "complete_episodes",
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
            'vf_share_layers': False,
            'use_lstm': True,
            'lstm_cell_size': 16,
            'max_seq_len': 5,
            'lstm_use_prev_action_reward': False,
            'framestack': False,
        },
        #"vf_loss_coeff": 0.000001,
        'multiagent': {
            'policies': {'agent_1': (None, obs_space, act_space, {}),
                         'agent_2': (None, obs_space, act_space, {}),
                         'agent_3': (None, obs_space, act_space, {}),
                         'agent_4': (None, obs_space, act_space, {}),
                         'agent_5': (None, obs_space, act_space, {})},
            'policy_mapping_fn': policy_mapping_fn,
        },
        "model": {
            "custom_model": "cc_model",
            "custom_model_config": {"num_agents": 5}
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
    base_config['env_config']['attack_randomization']['generator'] = 'UnbalancedAttackDefinitionGeneratorEvaluationRandom'
    base_config['evaluation_config']['env_config']['attack_randomization']['generator'] = 'UnbalancedAttackDefinitionGeneratorEvaluation'
    #base_config['env_config']['scenario_config']['custom_configs']['load_scaling_factor'] = 1.5
    #base_config['evaluation_config']['env_config']['scenario_config']['custom_configs']['load_scaling_factor'] = 1.5

    #base_config['env_config']['scenario_config']['custom_configs']['solar_scaling_factor'] = 3
    #base_config['evaluation_config']['env_config']['scenario_config']['custom_configs']['solar_scaling_factor'] = 3
    #base_config['env_config']['scenario_config']['custom_configs']['slack_bus_voltage'] = 1.04
    #base_config['evaluation_config']['env_config']['scenario_config']['custom_configs']['slack_bus_voltage'] = 1.04

    base_config['env_config']['M'] = 500
    base_config['env_config']['N'] = 0.3   #last
    base_config['env_config']['P'] = 2     #intt
    base_config['env_config']['Q'] = 1
    base_config['env_config']['T'] = 5

    if args.unbalance:
        for node in base_config['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['adversary_controller'] = 'adaptive_unbalanced_fixed_controller'
        for node in base_config['evaluation_config']['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['adversary_controller'] = 'adaptive_unbalanced_fixed_controller'
    else:
        for node in base_config['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['adversary_controller'] = 'oscillation_fixed_controller'
        for node in base_config['evaluation_config']['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['adversary_controller'] = 'oscillation_fixed_controller'

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
        run_hp_experiment(config, 'main')

    ray.shutdown()
