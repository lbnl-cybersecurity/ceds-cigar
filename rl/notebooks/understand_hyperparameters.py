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
import matplotlib.pyplot as plt
from collections import namedtuple

from pycigar.utils.input_parser import input_parser
from pycigar.utils.registry import make_create_env

ActionTuple = namedtuple('Action', ['action', 'timestep'])


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs per trial')
    parser.add_argument('--save-path', type=str, default='~/hp_exp', help='where to save the results')
    parser.add_argument('--workers', type=int, default=3, help='number of cpu workers per run')
    parser.add_argument('--eval-rounds', type=int, default=2,
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
            'avg_magnitude': None
        }
        self.rewards = []
        self.true_actions = [ActionTuple(2, -1)]  # 2 is the init action
        self.step = 0

    def update_sim_step(self, action, reward):
        """ Called after each simulation step """
        if action != self.true_actions[-1].action:
            self.true_actions.append(ActionTuple(action, self.step))

        self.rewards.append(reward)
        self.step += 1

    def update_sim_end(self, agent, env):
        """Called at the end of the simulation"""
        self['total_reward'] = sum(self.rewards)
        self['avg_magnitude'] = (np.array([t.action for t in self.true_actions[1:]]) - 2).mean()
        self['num_actions_taken'] = len(self.true_actions) - 1
        if len(self.true_actions) > 1:
            self['latest_action'] = self.true_actions[-1].timestep
            self['earliest_action'] = self.true_actions[1].timestep

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


def run_eval(agent, test_env, save_plot=None):
    """ Run one run of evaluation and return the stats """
    curr_stat = EpochStats()
    done = False
    obs = test_env.reset()
    while not done:
        act = agent.compute_action(obs)
        obs, r, done, _ = test_env.step(act)
        curr_stat.update_sim_step(act, r)

    curr_stat.update_sim_end(agent, test_env)

    if save_plot:
        f = test_env.plot(reward=curr_stat.stats['total_reward'])
        ax = f.axes
        ax[0].set_ylim([0.93, 1.07])
        ax[1].set_ylim([0, 5])
        ax[2].set_ylim([-280, 280])
        ax[3].set_ylim([0.91, 1.19])

        f.savefig(save_plot + '.png')

    return curr_stat


def run_train(config, reporter):
    create_env, env_name, create_test_env, test_env_name = make_create_env(params=config['pycigar_params'], version=0)
    register_env(env_name, create_env)
    register_env(test_env_name, create_test_env)
    test_env = create_test_env()
    obs_space = test_env.observation_space  # get the observation space, we need this to construct our agent(s) observation input
    act_space = test_env.action_space  # get the action space, we need this to construct our agent(s) action output

    agent = PPOTrainer(env=env_name, config=config['model_config'])

    for epoch in tqdm(range(config['epochs'])):
        result = agent.train()
        print(f'Running {config["eval_rounds"]} evaluation rounds')
        evals = []
        for i in range(config['eval_rounds']):
            if i == 0:
                res = run_eval(agent, test_env, save_plot=os.path.join(reporter.logdir, f'epoch{epoch}-test'))
            else:
                res = run_eval(agent, test_env)
            evals.append(res)

        eval_stat = EpochStats.average(evals)
        run_eval(agent, test_env, save_plot=os.path.join(reporter.logdir, f'epoch{epoch}-test'))
        reporter(**eval_stat.stats, **result)

    agent.stop()


def run_hp_experiment(full_config, name):
    print(full_config)
    res = tune.run(run_train,
                   config=full_config,
                   resources_per_trial={'cpu': 1, 'gpu': 0, 'extra_cpu': full_config['model_config']['num_workers']},
                   local_dir=os.path.join(os.path.expanduser(full_config['save_path']), name)
                   )
    with open(os.path.join(os.path.expanduser(full_config['save_path']), str(name) + '.pickle'), 'wb') as f:
        pickle.dump(res.trial_dataframes, f)


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
        'eval_rounds': args.eval_rounds,
        'save_path': args.save_path
    }

    config = deepcopy(full_config)
    config['pycigar_params']['sim_params']['N'] = ray.tune.grid_search([0, 1, 2, 4, 8])
    run_hp_experiment(config, 'action_penalty')

    config = deepcopy(full_config)
    config['pycigar_params']['sim_params']['P'] = ray.tune.grid_search([0, 1, 2, 4, 8])
    run_hp_experiment(config, 'init_penalty')

    config = deepcopy(full_config)
    config['model_config']['gamma'] = ray.tune.grid_search([0, 0.2, 0.5, 0.9, 1])
    run_hp_experiment(config, 'gamma')

    config = deepcopy(full_config)
    config['model_config']['lambda'] = ray.tune.grid_search([0, 0.2, 0.5, 0.9, 1])
    run_hp_experiment(config, 'lambda')

    config = deepcopy(full_config)
    config['model_config']['entropy_coeff'] = ray.tune.grid_search([0, 0.05, 0.1, 0.2, 0.5])
    run_hp_experiment(config, 'entropy_coeff')
