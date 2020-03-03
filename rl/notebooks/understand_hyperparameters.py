import argparse
import os
import pickle
from copy import deepcopy

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from tqdm import tqdm

from pycigar.utils.input_parser import input_parser
from pycigar.utils.registry import make_create_env


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs per run')
    parser.add_argument('--workers', type=int, default=3, help='number of cpu workers per run')
    parser.add_argument('--eval-rounds', type=int, default=4,
                        help='number of evaluation rounds to run to smooth random results')

    return parser.parse_args()


class EpochStats:
    """
    Statistics (on an evaluation run) about a given training
    """

    def __init__(self):
        self.stats = {
            'num_actions_taken': 0,
            'total_reward': 0,
            'earliest_action': None,
            'latest_action': None,
            'avg_entropy': None,
        }
        self.last_action = 2  # 2 is the init action
        self.rewards = []
        self.step = 0

    def update_sim_step(self, action, reward):
        """ Called after each simulation step """
        if self.stats['earliest_action'] is None and action != self.last_action:
            self.stats['earliest_action'] = self.step

        if action != self.last_action:
            self.stats['latest_action'] = self.step

        if action != self.last_action:
            self.last_action = action
            self.stats['num_actions_taken'] += 1

        self.rewards.append(reward)
        self.step += 1

    def update_sim_end(self, agent, env):
        """Called at the end of the simulation"""
        self.stats['total_reward'] = sum(self.rewards)

    @staticmethod
    def average(stats_list):
        """ Average the stats of many test runs """
        stat = EpochStats()
        for k in stat.stats:
            stat.stats[k] = np.mean([s.stats[k] for s in stats_list if s.stats[k] is not None])

        return stat

    def __setitem__(self, key, item):
        self.stats[key] = item

    def __getitem__(self, key):
        return self.stats[key]


def run_eval(agent, test_env):
    """ Run one run of evaluation and return the stats """
    curr_stat = EpochStats()
    done = False
    obs = test_env.reset()
    while not done:
        act = agent.compute_action(obs)
        obs, r, done, _ = test_env.step(act)
        curr_stat.update_sim_step(act, r)

    curr_stat.update_sim_end(agent, test_env)
    return curr_stat


def run_train(config, reporter):
    create_env, env_name, create_test_env, test_env_name = make_create_env(params=config['pycigar_params'], version=0)
    register_env(env_name, create_env)
    register_env(test_env_name, create_test_env)
    test_env = create_test_env()
    obs_space = test_env.observation_space  # get the observation space, we need this to construct our agent(s) observation input
    act_space = test_env.action_space  # get the action space, we need this to construct our agent(s) action output

    agent = PPOTrainer(env=env_name, config=config['model_config'])

    for _ in tqdm(range(config['epochs'])):
        result = agent.train()
        print(f'Running {config["eval_rounds"]} evaluation rounds')
        eval_stat = EpochStats.average([run_eval(agent, test_env) for _ in range(config['eval_rounds'])])
        reporter(**eval_stat.stats, **result)

    agent.stop()


def run_hp_experiment(full_config, name):
    print(full_config)
    res = tune.run(run_train,
                   config=full_config,
                   resources_per_trial={'cpu': 1, 'gpu': 0, 'extra_cpu': full_config['model_config']['num_workers']},
                   local_dir='/home/alexandre/hp_exp/' + name
                   )
    #with open('/home/alex/hp_exp/' + str(name) + '.pickle', 'wb') as f:
    #    pickle.dump(res.trial_dataframes, f)


sim_params = input_parser('ieee37busdata')
pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                  "env_name": "CentralControlPVInverterEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ['inverter_s701a']}

base_config = {
    "gamma": 0.5,
    'lr': 2e-04,
    'sample_batch_size': 50,
    'train_batch_size': 500,
    # 'lr_schedule': [[0, 5e-04], [12000, 5e-04], [13500, 5e-05]],

    'num_workers': 3,
    'num_cpus_per_worker': 1,
    'num_cpus_for_driver': 1,
    'num_envs_per_worker': 1,

    'log_level': 'ERROR',

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
}

if __name__ == '__main__':
    ray.init()
    args = parse_cli_args()

    base_config['num_workers'] = args.workers
    full_config = {
        'model_config': base_config,
        'pycigar_params': pycigar_params,
        'epochs': args.epochs,
        'eval_rounds': args.eval_rounds
    }


    config = deepcopy(full_config)
    config['pycigar_params']['N'] = ray.tune.grid_search([0, 1, 2, 4, 8])
    run_hp_experiment(config, 'action_penalty')

    config = deepcopy(full_config)
    config['pycigar_params']['P'] = ray.tune.grid_search([0, 1, 2, 4, 8])
    run_hp_experiment(config, 'init_penalty')
