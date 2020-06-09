import argparse
import os
import pickle
from copy import deepcopy

import numpy as np
import pycigar
import ray
from pycigar.utils.input_parser import input_parser
from pycigar.utils.registry import make_create_env
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from ray.tune.registry import register_env
from tqdm import tqdm
from pycigar.notebooks.utils import custom_eval_function, get_custom_callbacks, add_common_args

def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
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

    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
    base_config = {
        "env": env_name,
        "gamma": 0.5,
        'lr': 2e-4,
        'env_config': deepcopy(sim_params),
        'rollout_fragment_length': 50,
        'train_batch_size': 500,
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
            'fcnet_hiddens': [128, 64, 32],
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
        "evaluation_num_workers": 0 if args.local_mode else 1,
        'evaluation_num_episodes': args.eval_rounds,
        "evaluation_interval": args.eval_interval,
        "custom_eval_function": custom_eval_function,
        'evaluation_config': {
            "seed": 42,
            # IMPORTANT NOTE: For policy gradients, this might not be the optimal policy
            'explore': False,
            'env_config': deepcopy(sim_params),
        },

        # ==== CUSTOM METRICS ====
        "callbacks": get_custom_callbacks(),
    }
    # eval environment should not be random across workers
    eval_start = 100  # random.randint(0, 3599 - 500)
    base_config['evaluation_config']['env_config']['scenario_config']['start_end_time'] = [eval_start, eval_start + 750]
    base_config['evaluation_config']['env_config']['scenario_config']['multi_config'] = False
    del base_config['evaluation_config']['env_config']['attack_randomization']

    ray.init(local_mode=args.local_mode)

    full_config = {
        'config': base_config,
        'epochs': args.epochs,
        'save_path': args.save_path,
        'algo': args.algo,
    }

    full_config['config']['evaluation_config']['env_config']['M'] = tune.sample_from(
        lambda spec: np.random.choice([spec['config']['config']['env_config']['M']]))
    full_config['config']['evaluation_config']['env_config']['N'] = tune.sample_from(
        lambda spec: np.random.choice([spec['config']['config']['env_config']['N']]))
    full_config['config']['evaluation_config']['env_config']['P'] = tune.sample_from(
        lambda spec: np.random.choice([spec['config']['config']['env_config']['P']]))

    if args.algo == 'ppo':

        config = deepcopy(full_config)
        run_hp_experiment(config, 'main')

        """
        config = deepcopy(full_config)
        config['config']['env_config']['N'] = ray.tune.grid_search([0, 1, 2, 4, 8])
        run_hp_experiment(config, 'action_penalty')

        config = deepcopy(full_config)
        config['config']['env_config']['P'] = ray.tune.grid_search([0, 1, 2, 4, 8])
        run_hp_experiment(config, 'init_penalty')

        config = deepcopy(full_config)
        config['config']['env_config']['M'] = ray.tune.grid_search([0, 1, 2, 4, 8])
        run_hp_experiment(config, 'y_penalty')

        config = deepcopy(full_config)
        config['config']['gamma'] = ray.tune.grid_search([0, 0.2, 0.5, 0.9, 1])
        run_hp_experiment(config, 'gamma')

        config = deepcopy(full_config)
        config['config']['lambda'] = ray.tune.grid_search([0, 0.2, 0.5, 0.9, 1])
        run_hp_experiment(config, 'lambda')

        config = deepcopy(full_config)
        config['config']['entropy_coeff'] = ray.tune.grid_search([0, 0.05, 0.1, 0.2, 0.5])
        run_hp_experiment(config, 'entropy_coeff')

        config = deepcopy(full_config)
        config['config']['train_batch_size'] = ray.tune.grid_search([500, 1000, 2000, 4000])
        run_hp_experiment(config, 'batch_size')

        config = deepcopy(full_config)
        config['config']['lr'] = ray.tune.grid_search([2e-6, 2e-5, 2e-4, 2e-3, 2e-2])
        run_hp_experiment(config, 'lr')
        """
    elif args.algo == 'appo':
        config = deepcopy(full_config)
        run_hp_experiment(config, 'appo')

    ray.shutdown()
