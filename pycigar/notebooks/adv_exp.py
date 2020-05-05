import argparse
import random
from gym.spaces import Discrete

from ray import tune
from ray.rllib.agents.pg.pg import PGTrainer
from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf
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
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.tune.registry import register_env
from tqdm import tqdm

import pycigar
from pycigar.utils.input_parser import input_parser
from pycigar.utils.logging import logger
from pycigar.utils.output import plot_adv
from pycigar.utils.registry import make_create_env
from copy import copy


ActionTuple = namedtuple('Action', ['action', 'timestep'])

def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs for adversarial training PyCIGAR')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs per trial')
    parser.add_argument('--save-path', type=str, default='~/adv_exp', help='where to save the results')
    parser.add_argument('--workers', type=int, default=1, help='number of cpu workers per run')
    parser.add_argument('--eval-rounds', type=int, default=10,
                        help='number of evaluation rounds to run to smooth random results')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='do an evaluation every N epochs')
    parser.add_argument("--algo", help="use PPO or APPO", choices=['ppo', 'appo'],
                        nargs='?', const='ppo', default='ppo', type=str.lower)
    parser.add_argument('--adv', dest='adv', action='store_true')

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

    f = plot_adv(episodes[-1].hist_data['logger']['log_dict'], episodes[-1].hist_data['logger']['custom_metrics'], trainer.iteration, trainer.global_vars['adv'])
    f.savefig(trainer.global_vars['reporter_dir'] + 'eval-epoch-' + str(trainer.iteration) + '.png',
              bbox_inches='tight')

    save_best_policy(trainer, episodes)
    return metrics

# ==== CUSTOM METRICS (eval and training) ====
def on_episode_start(info):
    episode = info["episode"]

    # get the base env
    env = info['env'].get_unwrapped()[0]
    episode.user_data["tracking_id"] = env.k.device.get_rl_device_ids()[0]
    episode.user_data["hack_length"] = 0
    episode.user_data["defense_win"] = 0

def on_episode_step(info):
    episode = info["episode"]

    if episode.last_observation_for('attack_agent') is not None and episode._agent_reward_history['attack_agent']:
        if episode.user_data["hack_length"] != len(episode._agent_reward_history['attack_agent']):
            episode.user_data["hack_length"] = len(episode._agent_reward_history['attack_agent'])
            y = episode.last_observation_for('attack_agent')[1]
            # TODOs: check this condition for average y value
            if y < 0.07:
                episode.user_data["defense_win"] += 1


def on_episode_end(info):
    episode = info["episode"]

    tracking = logger()
    if episode.user_data["hack_length"] != 0:
        episode.hist_data['defense_win'] = [True] if episode.user_data['defense_win']/ float(episode.user_data['hack_length']) > 0.6 else [False] # defense agent win 60% of the hack period
    episode.hist_data['logger'] = {'log_dict': tracking.log_dict, 'custom_metrics': tracking.custom_metrics}

def save_best_policy(trainer, episodes):
    train_policy = trainer.config['multiagent']['policies_to_train'][0]
    mean_r = np.array([ep.agent_rewards[key] for ep in episodes for key in ep.agent_rewards.keys() if train_policy in key]).mean()
    if 'best_eval_reward' not in trainer.global_vars or trainer.global_vars['best_eval_reward'] < mean_r:
        os.makedirs(os.path.join(trainer.global_vars['reporter_dir'], 'best'), exist_ok=True)
        trainer.global_vars['best_eval_reward'] = mean_r
        # save policy
        shutil.rmtree(os.path.join(trainer.global_vars['reporter_dir'], 'best', train_policy + '_policy'), ignore_errors=True)
        trainer.get_policy(train_policy).export_model(os.path.join(trainer.global_vars['reporter_dir'], 'best', train_policy + '_policy'))

        # save plots
        ep = episodes[-1]
        data = ep.hist_data['logger']['log_dict']
        f = plot_adv(data, ep.hist_data['logger']['custom_metrics'], trainer.iteration, trainer.global_vars['adv'])
        f.savefig(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'eval.png'))

        # save CSV
        """
        k = list(data.keys())[0]
        ep_hist = pd.DataFrame(dict(v=data[data[k]['node']]['voltage'], y=data[k]['y'],
                                    q_set=data[k]['q_set'], q_val=data[k]['q_out']))
        a_hist = pd.DataFrame(data[k]['control_setting'], columns=['a1', 'a2', 'a3', 'a4', 'a5'])
        adv_a_hist = pd.DataFrame(data['adversary_' + k]['control_setting'],
                                  columns=['adv_a1', 'adv_a2', 'adv_a3', 'adv_a4', 'adv_a5'])
        translation, slope = get_translation_and_slope(data[k]['control_setting'])
        adv_translation, adv_slope = get_translation_and_slope(data['adversary_' + k]['control_setting'])
        trans_slope_hist = pd.DataFrame(dict(translation=translation, slope=slope,
                                             adv_translation=adv_translation, adv_slope=adv_slope))

        df = ep_hist.join(a_hist, how='outer')
        df = df.join(adv_a_hist, how='outer')
        df = df.join(trans_slope_hist, how='outer')
        df.to_csv(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'last_eval_hists.csv'))
        """
        # save info
        #start = ep.custom_metrics["hack_start"]
        #end = ep.custom_metrics["hack_end"]
        info = {
            'epoch': trainer.iteration,
        #    'hack_start': start,
        #    'hack_end': end,
            'reward': mean_r
        }
        with open(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)


class ConstantPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.exploration = self._create_exploration()

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):

        return [(1,2,3,4,5) for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

def select_policy(agent_id):
    if agent_id == "defense_agent":
        return "defense"
    else:
        return "attack"

def run_train(config, reporter):
    if not config['adv']:
        trainer_cls = APPOTrainer if config['algo'] == 'appo' else PPOTrainer
        config['config']['multiagent']['policies']['defense'] = (None,
                                                                 config['config']['multiagent']['policies']['defense'][1],
                                                                 config['config']['multiagent']['policies']['defense'][2],
                                                                 {})
        config['config']['multiagent']['policies_to_train'] = ['defense']
        trainer = trainer_cls(config=config['config'])

        # needed so that the custom eval fn knows where to save plots
        trainer.global_vars['reporter_dir'] = reporter.logdir
        trainer.global_vars['adv'] = config['adv']

        for _ in tqdm(range(config['epochs'])):
            results = trainer.train()
            del results['hist_stats']['logger']  # don't send to tensorboard
            if 'evaluation' in results:
                del results['evaluation']['hist_stats']['logger']
            reporter(**results)

        trainer.stop()

    else:
        trainer_cls = APPOTrainer if config['algo'] == 'appo' else PPOTrainer
        defense_config = copy(config)
        defense_config['config']['multiagent']['policies_to_train'] = ['defense']
        attack_config = copy(config)
        attack_config['config']['multiagent']['policies_to_train'] = ['attack']

        defense_trainer = trainer_cls(config=defense_config['config'])
        attack_trainer = trainer_cls(config=attack_config['config'])

        defense_trainer.global_vars['reporter_dir']= reporter.logdir
        defense_trainer.global_vars['adv'] = config['adv']
        attack_trainer.global_vars['reporter_dir']= reporter.logdir
        attack_trainer.global_vars['adv'] = config['adv']

        for i in tqdm(range(config['epochs'])):
            if i == 0:
                # load pretrained defense_trainer
                pass

            while True:
                attack_trainer.get_policy('defense').set_weights(attack_trainer.get_policy('defense').get_weights())
                results = attack_trainer.train()
                evaluate = attack_trainer._evaluate()
                num_lose = sum(evaluate['evaluation']['hist_stats']['defense_win'])
                num_match = len(evaluate['evaluation']['hist_stats']['defense_win'])
                if (num_match - num_lose)/float(num_match) >=0.6:
                    del results['hist_stats']['logger']
                    if 'evaluation' in results:
                        del results['evaluation']['hist_stats']['logger']
                    reporter(**results)
                    break
            while True:
                defense_trainer.get_policy('attack').set_weights(attack_trainer.get_policy('attack').get_weights())
                results = defense_trainer.train()
                evaluate = defense_trainer._evaluate()
                num_win = sum(evaluate['evaluation']['hist_stats']['defense_win'])
                num_match = len(evaluate['evaluation']['hist_stats']['defense_win'])
                if num_win/float(num_match) >=0.6:
                    del results['hist_stats']['logger']
                    if 'evaluation' in results:
                        del results['evaluation']['hist_stats']['logger']
                    reporter(**results)
                    break

        defense_trainer.stop()
        attack_trainer.stop()

def run_defense_vs_attack(full_config, name):

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

if __name__ == "__main__":
    args = parse_cli_args()

    pycigar_params = {'exp_tag': 'adversarial_training',
                      'env_name': 'AdvMultiEnv',
                      'simulator': 'opendss'}

    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, adv=True)

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)
    test_env = create_env(sim_params)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    base_config = {
        "env": env_name,
        'env_config': deepcopy(sim_params),
        'num_workers': args.workers,
        'num_cpus_per_worker': 1,
        'num_cpus_for_driver': 1,
        'num_gpus': 0,
        'num_gpus_per_worker': 0,
        'num_envs_per_worker': 1,
        'log_level': 'WARNING',
        "gamma": 0.5,
        'lr': 2e-4,
        'no_done_at_end': False,
        'rollout_fragment_length': 50,
        'train_batch_size': 200,
        'clip_param': 0.1,
        'lambda': 0.95,
        'vf_clip_param': 100,
        "multiagent": {
            "policies_to_train": ["attack", "defense"],
            "policies": {
                "attack": (None,
                            obs_space,
                            act_space,
                            {
                    'model': {
                        'fcnet_activation': 'tanh',
                        'fcnet_hiddens': [256, 128, 128, 64],
                        'free_log_std': False,
                        'vf_share_layers': True,
                        'use_lstm': False,
                        'state_shape': None,
                        'framestack': False,
                        'zero_mean': True,
                    },
                            }),
                "defense": (None,
                           obs_space,
                           act_space,
                           {
                        'model': {
                        'fcnet_activation': 'tanh',
                        'fcnet_hiddens': [256, 128, 128, 64],
                        'free_log_std': False,
                        'vf_share_layers': True,
                        'use_lstm': False,
                        'state_shape': None,
                        'framestack': False,
                        'zero_mean': True,
                    },
                           }),
            },
            "policy_mapping_fn": select_policy,
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
            'env_config': deepcopy(sim_params),
        },

        # ==== CUSTOM METRICS ====
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
        },
    }
    # eval environment should not be random across workers
    eval_start = 100  # random.randint(0, 3599 - 500)
    base_config['evaluation_config']['env_config']['scenario_config']['start_end_time'] = [eval_start, eval_start + 750]
    base_config['evaluation_config']['env_config']['scenario_config']['multi_config'] = False
    del base_config['evaluation_config']['env_config']['attack_randomization']

    ray.init(local_mode=True)
    full_config = {
        'config': base_config,
        'epochs': args.epochs,
        'save_path': args.save_path,
        'algo': args.algo,
        'adv': args.adv
    }

    if args.algo == 'ppo':
        config = deepcopy(full_config)
        run_defense_vs_attack(config, 'attack_train')