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
from pycigar.notebooks.utils import custom_eval_function, add_common_args, CustomCallbacks
from pycigar.utils.input_parser import input_parser
from pycigar.utils.logging import logger
from pycigar.utils.output import plot_new
from pycigar.utils.registry import make_create_env

ActionTuple = namedtuple('Action', ['action', 'timestep'])


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Experimentations of the base attack')
    add_common_args(parser)

    return parser.parse_args()


def run_train(config, reporter):
    trainer_cls = APPOTrainer if config['algo'] == 'appo' else PPOTrainer
    trainer = trainer_cls(config=config['config'])

    # needed so that the custom eval fn knows where to save plots
    trainer.global_vars['reporter_dir'] = reporter.logdir
    trainer.global_vars['unbalance'] = False

    for _ in tqdm(range(config['epochs'])):
        results = trainer.train()
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
                      'env_name': 'CentralControlPVInverterEnv',
                      'simulator': 'opendss'}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)

    misc_inputs_path = pycigar.DATA_DIR + "/ieee240busdata/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee240busdata/Master.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee240busdata/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee240busdata/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
    sim_params['vectorized_mode'] = True
    sim_params['is_disable_log'] = True
    sim_params_eval = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
    sim_params_eval['vectorized_mode'] = True
    sim_params_eval['is_disable_log'] = False

    base_config = {
        "env": env_name,
        "gamma": 0.5,
        'lr': 2e-4,
        'env_config': deepcopy(sim_params),
        'rollout_fragment_length': 20,
        'train_batch_size': 128, #min(500, 20*args.workers),
        'sgd_minibatch_size': 20,
        'clip_param': 0.1,
        'lambda': 0.95,
        'vf_clip_param': 100,

        'num_workers': 1, #args.workers,
        'num_cpus_per_worker': 1,
        'num_cpus_for_driver': 1,
        'num_envs_per_worker': 1,

        'log_level': 'WARNING',

        'model': {
            'fcnet_activation': 'tanh',
            'fcnet_hiddens': [32, 32],
            'free_log_std': False,
            'vf_share_layers': True,
            'use_lstm': False,
            'state_shape': None,
            'framestack': False,
            'zero_mean': True,
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
        "callbacks": CustomCallbacks,
    }
    # eval environment should not be random across workers
    base_config['env_config']['attack_randomization']['generator'] = 'AttackDefinitionGeneratorEvaluationRandom'
    base_config['evaluation_config']['env_config']['attack_randomization']['generator'] = 'AttackDefinitionGeneratorEvaluation'

    ray.init(local_mode=True)

    full_config = {
        'config': base_config,
        'epochs': args.epochs,
        'save_path': args.save_path,
        'algo': args.algo,
    }

    config = deepcopy(full_config)
    run_hp_experiment(config, 'main')

    ray.shutdown()
