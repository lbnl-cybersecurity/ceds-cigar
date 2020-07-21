import argparse
import os
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pycigar
import ray
from pycigar.notebooks.utils import add_common_args, get_base_config
from pycigar.utils.input_parser import input_parser
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from tqdm import tqdm


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
    res = tune.run(
        run_train,
        config=full_config,
        resources_per_trial={
            'cpu': 1,
            'gpu': 0,
            'extra_cpu': full_config['config']['num_workers'] + full_config['config']['evaluation_num_workers'],
        },
        local_dir=os.path.join(os.path.expanduser(full_config['save_path']), name),
    )
    # save results
    with open(os.path.join(os.path.expanduser(full_config['save_path']), str(name) + '.pickle'), 'wb') as f:
        pickle.dump(res.trial_dataframes, f)


if __name__ == '__main__':
    args = parse_cli_args()
    feeder_path = Path(pycigar.DATA_DIR) / 'ieee37busdata'

    sim_params = input_parser(misc_inputs_path=str(feeder_path / 'misc_inputs.csv'),
                              dss_path=str(feeder_path / 'ieee37.dss'),
                              load_solar_path=str(feeder_path / 'load_solar_data.csv'),
                              breakpoints_path=str(feeder_path / 'breakpoints.csv'))
    base_config, create_env = \
        get_base_config(env_name='CentralControlPVInverterEnv',
                        cli_args=args,
                        sim_params=sim_params)

    if args.redis_pwd:
        # for running in a cluster
        ray.init(address='auto', redis_password=args.redis_pwd)
        print("Nodes in the Ray cluster:")
        print(ray.nodes())
    else:
        ray.init(local_mode=args.local_mode)

    full_config = {
        'config': base_config,
        'epochs': args.epochs,
        'save_path': args.save_path,
        'algo': args.algo,
    }

    full_config['config']['evaluation_config']['env_config']['M'] = tune.sample_from(
        lambda spec: np.random.choice([spec['config']['config']['env_config']['M']])
    )
    full_config['config']['evaluation_config']['env_config']['N'] = tune.sample_from(
        lambda spec: np.random.choice([spec['config']['config']['env_config']['N']])
    )
    full_config['config']['evaluation_config']['env_config']['P'] = tune.sample_from(
        lambda spec: np.random.choice([spec['config']['config']['env_config']['P']])
    )

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
