import argparse
import os
import pickle
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycigar
import ray
import tensorflow as tf
from pycigar.notebooks.utils import get_base_config, add_common_args
from pycigar.utils.input_parser import input_parser
from pycigar.utils.logging import logger
from pycigar.utils.output import plot_new
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from tqdm import tqdm


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Experimentations of the unbalance attack')
    parser.add_argument('--eval-saved', type=str, default='', help='eval a trained agent')
    parser.add_argument('--no-attack', dest='noattack', action='store_true')
    parser.add_argument('--continuous', action='store_true')

    add_common_args(parser)

    return parser.parse_args()


def run_train(config, reporter):
    trainer_cls = APPOTrainer if config['algo'] == 'appo' else PPOTrainer
    trainer = trainer_cls(config=config['config'])

    # needed so that the custom eval fn knows where to save plots
    trainer.global_vars['reporter_dir'] = reporter.logdir
    trainer.global_vars['unbalance'] = True  # for plots

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


def set_unbalance_attack(base_config):
    for node in base_config['env_config']['scenario_config']['nodes']:
        for d in node['devices']:
            d['adversary_controller'] = 'unbalanced_fixed_controller'
    for node in base_config['evaluation_config']['env_config']['scenario_config']['nodes']:
        for d in node['devices']:
            d['adversary_controller'] = 'unbalanced_fixed_controller'


def adjust_default_curves(base_config):
    for node in (
            base_config['env_config']['scenario_config']['nodes']
            + base_config['evaluation_config']['env_config']['scenario_config']['nodes']
    ):
        for d in node['devices']:
            name = d['name']
            c = np.array(d['custom_configs']['default_control_setting'])
            # found by training with no attack
            if name.endswith('a'):
                c = c - 0.02
            elif name.endswith('b'):
                c = c + 0.02
            elif name.endswith('c'):
                c = c - 0.01
            d['custom_configs']['default_control_setting'] = c


if __name__ == '__main__':
    args = parse_cli_args()
    feeder_path = Path(pycigar.DATA_DIR) / 'ieee37busdata_regulator_attack'

    sim_params = input_parser(misc_inputs_path=str(feeder_path / 'misc_inputs.csv'),
                              dss_path=str(feeder_path / 'ieee37.dss'),
                              load_solar_path=str(feeder_path / 'load_solar_data.csv'),
                              breakpoints_path=str(feeder_path / 'breakpoints.csv'),
                              percentage_hack=0.3)
    base_config, create_env = \
        get_base_config(env_name=f'CentralControlPhaseSpecific{"Continuous" if args.continuous else ""}PVInverterEnv',
                        cli_args=args,
                        sim_params=sim_params)

    set_unbalance_attack(base_config)
    #adjust_default_curves(base_config)

    if args.noattack:
        for node in base_config['env_config']['scenario_config']['nodes'] \
                    + base_config['evaluation_config']['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['hack'] = [50, 0, 50]

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

    if args.eval_saved:
        test_env = create_env(full_config['config']['evaluation_config']['env_config'])
        tf.compat.v1.enable_eager_execution()
        policy = tf.saved_model.load(os.path.expanduser(args.eval_saved))
        infer = policy.signatures['serving_default']
        done = False
        obs = test_env.reset()
        obs = obs.tolist()
        while not done:
            act_logits = infer(
                prev_reward=tf.constant([0.0], tf.float32),
                observations=tf.constant([obs], tf.float32),
                is_training=tf.constant(False),
                seq_lens=tf.constant([0], tf.int32),
                prev_action=tf.constant([0], tf.int64),
            )['behaviour_logits'].numpy()
            act = np.argmax(np.stack(np.array_split(act_logits[0], 3)), axis=1)
            print(act)
            obs, r, done, _ = test_env.step(act)
            obs = obs.tolist()

        Logger = logger()
        f = plot_new(Logger.log_dict, Logger.custom_metrics, None, unbalance=True)
        f.savefig('eval.png', bbox_inches='tight')
        plt.close(f)

    else:
        full_config['config']['evaluation_config']['env_config']['M'] = tune.sample_from(
            lambda spec: np.random.choice([spec['config']['config']['env_config']['M']])
        )
        full_config['config']['evaluation_config']['env_config']['N'] = tune.sample_from(
            lambda spec: np.random.choice([spec['config']['config']['env_config']['N']])
        )
        full_config['config']['evaluation_config']['env_config']['P'] = tune.sample_from(
            lambda spec: np.random.choice([spec['config']['config']['env_config']['P']])
        )

        config = deepcopy(full_config)
        config['config']['env_config']['simulation_config']['network_model_directory'] = [str(feeder_path / 'ieee37.dss'), str(feeder_path / 'ieee37_b.dss'), str(feeder_path / 'ieee37_c.dss')]
        config['config']['env_config']['env_config']['sims_per_step'] = 50
        config['config']['evaluation_config']['env_config']['env_config']['sims_per_step'] = 50
        config['config']['env_config']['is_disable_y'] = True
        config['config']['evaluation_config']['is_disable_y'] = True
        config['config']['env_config']['M'] = 50000
        config['config']['env_config']['N'] = tune.grid_search([30, 40])
        config['config']['env_config']['P'] = 100
        config['config']['env_config']['T'] = 0
        config['config']['lr'] = 1e-4
        config['config']['clip_param'] = 0.1

        run_hp_experiment(config, 'main')

    ray.shutdown()
