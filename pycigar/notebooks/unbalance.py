import argparse
import os
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pycigar
import ray
import tensorflow as tf
from pycigar.notebooks.utils import custom_eval_function, get_custom_callbacks, add_common_args
from pycigar.utils.input_parser import input_parser
from pycigar.utils.logging import logger
from pycigar.utils.output import plot_new
from pycigar.utils.registry import make_create_env
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from ray.tune.registry import register_env
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
                      'env_name': f'CentralControlPhaseSpecific{"Continuous" if args.continuous else ""}PVInverterEnv',
                      'simulator': 'opendss'}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)

    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata_regulator_attack/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
    base_config = {
        "env": env_name,
        "gamma": 0.5,
        'lr': 2e-3,
        'env_config': deepcopy(sim_params),
        'rollout_fragment_length': 35,
        'train_batch_size': max(500, 35 * args.workers),
        'clip_param': 0.15,
        'lambda': 0.95,
        'vf_clip_param': 10000,

        'num_workers': args.workers,
        'num_cpus_per_worker': 1,
        'num_cpus_for_driver': 1,
        'num_envs_per_worker': 1,

        'log_level': 'WARNING',

        'model': {
            'fcnet_activation': 'tanh',
            'fcnet_hiddens': [64, 64, 32],
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
    base_config['evaluation_config']['env_config']['scenario_config']['start_end_time'] = [eval_start,
                                                                                           eval_start + 750]
    base_config['evaluation_config']['env_config']['scenario_config']['multi_config'] = False
    del base_config['evaluation_config']['env_config']['attack_randomization']
    del base_config['env_config']['attack_randomization']

    for node in base_config['env_config']['scenario_config']['nodes']:
        for d in node['devices']:
            d['adversary_controller'] = 'unbalanced_fixed_controller'
    for node in base_config['evaluation_config']['env_config']['scenario_config']['nodes']:
        for d in node['devices']:
            d['adversary_controller'] = 'unbalanced_fixed_controller'

    if args.noattack:
        for node in base_config['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['hack'] = [50, 0, 50]
        for node in base_config['evaluation_config']['env_config']['scenario_config']['nodes']:
            for d in node['devices']:
                d['hack'] = [50, 0, 50]

    for node in base_config['env_config']['scenario_config']['nodes'] \
                + base_config['evaluation_config']['env_config']['scenario_config']['nodes']:
        for d in node['devices']:
            name = d['name']
            c = np.array(d['custom_configs']['default_control_setting'])
            #            found by training with no attack
            if name.endswith('a'):
                c = c - 0.02
            elif name.endswith('b'):
                c = c + 0.02
            elif name.endswith('c'):
                c = c - 0.01
            d['custom_configs']['default_control_setting'] = c

    if args.redis_pwd:
        ray.init(address='auto', redis_password=args.redis_pwd)
    else:
        ray.init(local_mode=args.local_mode)

    print("Nodes in the Ray cluster:")
    print(ray.nodes())

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
                prev_reward=tf.constant([0.], tf.float32),
                observations=tf.constant([obs], tf.float32),
                is_training=tf.constant(False),
                seq_lens=tf.constant([0], tf.int32),
                prev_action=tf.constant([0], tf.int64)
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
            lambda spec: np.random.choice([spec['config']['config']['env_config']['M']]))
        full_config['config']['evaluation_config']['env_config']['N'] = tune.sample_from(
            lambda spec: np.random.choice([spec['config']['config']['env_config']['N']]))
        full_config['config']['evaluation_config']['env_config']['P'] = tune.sample_from(
            lambda spec: np.random.choice([spec['config']['config']['env_config']['P']]))

        config = deepcopy(full_config)
        config['config']['env_config']['M'] = 50000
        config['config']['env_config']['N'] = 30
        config['config']['env_config']['P'] = 0
        config['config']['lr'] = 1e-3
        config['config']['clip_param'] = 0.15

        run_hp_experiment(config, 'main_N30')

    ray.shutdown()
