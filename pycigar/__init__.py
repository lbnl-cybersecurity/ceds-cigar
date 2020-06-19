"""Returns features of the PyCIGAR repository (e.g. version number)."""

# flow repo version number
__version__ = "1.0.0"


import argparse
import math
import os
import json
import ray
from ray import tune
from pycigar.utils.input_parser import input_parser
from pycigar.utils.output import pycigar_output_specs
from pycigar.utils.registry import make_create_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from tqdm import tqdm
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
import numpy as np
import shutil
from copy import deepcopy
import tensorflow as tf
from pycigar.config import *


def main(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, test, policy, output):
    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
    pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                      'env_name': 'CentralControlPVInverterEnv',
                      'simulator': 'opendss'}
    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)

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

        save_best_policy(trainer, episodes)
        return metrics

    def save_best_policy(trainer, episodes):
        mean_r = np.array([ep.episode_reward for ep in episodes]).mean()
        if 'best_eval_reward' not in trainer.global_vars or trainer.global_vars['best_eval_reward'] < mean_r:
            os.makedirs(output, exist_ok=True)
            trainer.global_vars['best_eval_reward'] = mean_r
            # save policy
            shutil.rmtree(os.path.join(output, 'policy'), ignore_errors=True)
            trainer.get_policy().export_model(os.path.join(output, 'policy'))
            # save info
            info = {
                'epoch': trainer.iteration,
                'reward': mean_r
            }
            with open(os.path.join(output, 'info.json'), 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=4)

    def run_train(config, reporter):
        trainer_cls = PPOTrainer
        trainer = trainer_cls(config=config['config'])

        # needed so that the custom eval fn knows where to save plots
        trainer.global_vars['reporter_dir'] = reporter.logdir
        for _ in tqdm(range(config['epochs'])):
            results = trainer.train()
            reporter(**results)
        trainer.stop()

    def run_experiment(full_config):
        res = tune.run(run_train,
                       config=full_config,
                       resources_per_trial={'cpu': 1, 'gpu': 0,
                                        'extra_cpu': full_config['config']['num_workers']
                                        + full_config['config']['evaluation_num_workers']},
                       local_dir=os.path.expanduser(full_config['save_path']))

    base_config = {
        "env": env_name,
        "gamma": 0.5,
        'lr': 2e-4,
        'env_config': deepcopy(sim_params),
        'sample_batch_size': 50,
        'train_batch_size': 500,
        'clip_param': 0.1,
        'lambda': 0.95,
        'vf_clip_param': 100,

        'num_workers': 5,
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
        "evaluation_num_workers": 1,
        'evaluation_num_episodes': 2,
        "evaluation_interval": 1,
        "custom_eval_function": custom_eval_function,
        'evaluation_config': {
            "seed": 42,
            # IMPORTANT NOTE: For policy gradients, this might not be the optimal policy
            'explore': False,
            'env_config': deepcopy(sim_params),
        },
    }
    eval_start = 100
    base_config['evaluation_config']['env_config']['scenario_config']['start_end_time'] = [eval_start, eval_start + 750]
    del base_config['evaluation_config']['env_config']['attack_randomization']
    base_config['evaluation_config']['env_config']['scenario_config']['multi_config'] = False
    test_env = create_env(base_config['evaluation_config']['env_config'])
    test_env.observation_space
    test_env.action_space

    if test == 0:
        ray.init()
        base_config['num_workers'] = 1
        base_config['evaluation_num_episodes'] = 2
        full_config = {
            'config': base_config,
            'epochs': 1,
            'save_path': output,
        }
        config = deepcopy(full_config)
        run_experiment(config)

    elif test == 1:
        # Test using the provided trained agent
        tf.compat.v1.enable_eager_execution()
        policy = tf.saved_model.load(policy)
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
            act = np.argmax(act_logits)
            obs, r, done, _ = test_env.step(act)
            obs = obs.tolist()

        output_specs = pycigar_output_specs(test_env)
        if not os.path.exists(output):
            os.mkdir(output)
        with open(os.path.join(output, 'pycigar_output_specs.json'), 'w') as outfile:
            json.dump(output_specs, outfile, indent=4)

    else:
        obs = test_env.reset()
        done = False
        while not done:
            # for each observation, let the policy decides what to do
            obs, r, done, _ = test_env.step(2)

        output_specs = pycigar_output_specs(test_env)
        if not os.path.exists(output):
            os.mkdir(output)
        with open(os.path.join(output, 'pycigar_output_specs.json'), 'w') as outfile:
            json.dump(output_specs, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=os.path.abspath, default=DATA_DIR + "/ieee37busdata/misc_inputs.csv", help='Directory to params.csv file.')
    parser.add_argument("--dss", type=os.path.abspath, default=DATA_DIR + "/ieee37busdata/ieee37.dss", help='Directory to .dss file.')
    parser.add_argument("--loadpv", type=os.path.abspath, default=DATA_DIR + "/ieee37busdata/load_solar_data.csv", help='Directory to load-solar.csv file.')
    parser.add_argument("--breakpoints", type=os.path.abspath, default=None, help='Directory to custom initial breakpoints.csv file (Optional)')
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--policy", type=os.path.abspath, default=LOG_DIR + "/policy/")
    parser.add_argument("--output", type=os.path.abspath, default=LOG_DIR)
    args = parser.parse_args()
    # Run
    main(args.params, args.dss, args.loadpv, args.breakpoints, args.test, args.policy, args.output)