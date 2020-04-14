import argparse
import math
import os
import pickle
import json
import shutil
import random
from collections import namedtuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.tune.registry import register_env
from tqdm import tqdm

from pycigar.utils.input_parser import input_parser
from pycigar.utils.registry import make_create_env

ActionTuple = namedtuple('Action', ['action', 'timestep'])


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs per trial')
    parser.add_argument('--save-path', type=str, default='~/hp_experiment3', help='where to save the results')
    parser.add_argument('--workers', type=int, default=3, help='number of cpu workers per run')
    parser.add_argument('--eval-rounds', type=int, default=2,
                        help='number of evaluation rounds to run to smooth random results')
    parser.add_argument("--algo", help="use PPO or APPO", choices=['ppo', 'appo'],
                        nargs='?', const='ppo', default='ppo', type=str.lower)

    return parser.parse_args()


def custom_eval_function(trainer, eval_workers):
    # same behaviour as
    # https://github.com/ray-project/ray/blob/7ebc6783e4a1f6b32753c35cea70973763c996f1/rllib/agents/trainer.py#L707-L723
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

    f = plot_episode(episodes[-1], trainer.iteration)
    f.savefig(trainer.global_vars['reporter_dir'] + 'eval-epoch-' + str(trainer.iteration) + '.png', bbox_inches='tight')

    save_best_policy(trainer, episodes)

    return metrics


# ==== CUSTOM METRICS (eval and training) ====
def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["num_actions_taken"] = 0
    episode.user_data["magnitudes"] = []
    episode.user_data["true_actions"] = [ActionTuple(2, -1)]  # 2 is the init action
    episode.hist_data["y"] = []

    # get the base env
    env = info['env'].get_unwrapped()[0]
    while not hasattr(env, 'tracking_ids'):
        env = env.env
    episode.user_data["tracking_id"] = env.tracking_ids[0]


def on_episode_step(info):
    episode = info["episode"]
    action = episode.last_action_for()
    if action != episode.user_data["true_actions"][-1].action:
        episode.user_data["true_actions"].append(ActionTuple(action, episode.length))

    if episode.last_info_for() is not None:
        y = episode.last_info_for()[episode.user_data["tracking_id"]]['y']
        episode.hist_data["y"].append(y)


def on_episode_end(info):
    episode = info["episode"]
    actions = episode.user_data["true_actions"]
    avg_mag = (np.array([t.action for t in actions[1:]]) - 2).mean()
    num_actions = len(actions) - 1
    if num_actions > 0:
        episode.custom_metrics['latest_action'] = actions[-1].timestep
        episode.custom_metrics['earliest_action'] = actions[1].timestep

    episode.custom_metrics["avg_magnitude"] = avg_mag
    episode.custom_metrics["num_actions_taken"] = num_actions

    env = info['env'].vector_env.envs[0]
    t_id = env.unwrapped.tracking_ids[0]
    episode.hist_data.update(env.unwrapped.tracking_infos[t_id])
    episode.hist_data['adversary_a_val'] = env.unwrapped.tracking_infos['adversary_' + t_id]['a_val']
    hack_start = int([k for k, v in env.unwrapped.k.scenario.hack_start_times.items() if 'adversary_' + t_id in v][0])
    hack_end = int([k for k, v in env.unwrapped.k.scenario.hack_end_times.items() if 'adversary_' + t_id in v][0])
    episode.custom_metrics["hack_start"] = hack_start
    episode.custom_metrics["hack_end"] = hack_end


# ==== ====
def get_translation_and_slope(a_val):
    points = np.array(a_val)
    slope = points[:, 1] - points[:, 0]
    og_point = points[0, 2]
    translation = points[:, 2] - og_point
    return translation, slope


def plot_episode(ep, epoch=None):
    plt.rc('font', size=15)
    plt.rc('figure', titlesize=35)
    f, ax = plt.subplots(5, figsize=(25, 20))
    title = '[epoch {}] total reward: {:.2f}'.format(epoch, ep.episode_reward)
    f.suptitle(title)
    ax[0].plot(ep.hist_data['v_val'], color='tab:blue', label='voltage')

    ax[1].plot(ep.hist_data['y_val'], color='tab:blue', label='oscillation observer')

    ax[2].plot(ep.hist_data['q_set'], color='tab:blue', label='q_set')
    ax[2].plot(ep.hist_data['q_val'], color='tab:orange', label='q_val')

    translation, slope = get_translation_and_slope(ep.hist_data['a_val'])
    ax[3].plot(translation, color='tab:blue', label='RL translation')
    ax[3].plot(slope, color='tab:purple', label='RL slope (a2-a1)')

    translation, slope = get_translation_and_slope(ep.hist_data['adversary_a_val'])
    ax[4].plot(translation, color='tab:orange', label='hacked translation')
    ax[4].plot(slope, color='tab:red', label='hacked slope (a2-a1)')
    ax[0].set_ylim([0.93, 1.07])
    ax[1].set_ylim([0, 0.8])
    ax[2].set_ylim([-280, 280])
    ax[3].set_ylim([-0.055, 0.055])
    ax[4].set_ylim([-0.055, 0.055])

    for a in ax:
        a.grid(b=True, which='both')
        a.legend(loc=1, ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return f


def save_best_policy(trainer, episodes):
    mean_r = np.array([ep.episode_reward for ep in episodes]).mean()
    if 'best_eval_reward' not in trainer.global_vars or trainer.global_vars['best_eval_reward'] < mean_r:
        os.makedirs(os.path.join(trainer.global_vars['reporter_dir'], 'best'), exist_ok=True)
        trainer.global_vars['best_eval_reward'] = mean_r
        # save policy
        shutil.rmtree(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'policy'), ignore_errors=True)
        trainer.get_policy().export_model(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'policy'))
        # save plots
        ep = episodes[-1]
        f = plot_episode(ep, trainer.iteration)
        f.savefig(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'eval.png'))

        # save CSV
        ep_hist = pd.DataFrame(dict(v=ep.hist_data['v_val'], y=ep.hist_data['y_val'],
                                    q_set=ep.hist_data['q_set'], q_val=ep.hist_data['q_val']))
        a_hist = pd.DataFrame(ep.hist_data['a_val'], columns=['a1', 'a2', 'a3', 'a4', 'a5'])
        adv_a_hist = pd.DataFrame(ep.hist_data['a_val'], columns=['adv_a1', 'adv_a2', 'adv_a3', 'adv_a4', 'adv_a5'])
        translation, slope = get_translation_and_slope(ep.hist_data['a_val'])
        adv_translation, adv_slope = get_translation_and_slope(ep.hist_data['adversary_a_val'])
        trans_slope_hist = pd.DataFrame(dict(translation=translation, slope=slope,
                                             adv_translation=adv_translation, adv_slope=adv_slope))

        df = ep_hist.join(a_hist, how='outer')
        df = df.join(adv_a_hist, how='outer')
        df = df.join(trans_slope_hist, how='outer')
        df.to_csv(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'last_eval_hists.csv'))

        # save info
        start = ep.custom_metrics["hack_start"]
        end = ep.custom_metrics["hack_end"]
        info = {
            'epoch': trainer.iteration,
            'hack_start': start,
            'hack_end': end,
            'reward': mean_r
        }
        with open(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)


def run_train(config, reporter):
    trainer_cls = APPOTrainer if config['algo'] == 'appo' else PPOTrainer
    trainer = trainer_cls(config=config['config'])

    # needed so that the custom eval fn knows where to save plots
    trainer.global_vars['reporter_dir'] = reporter.logdir
    for _ in tqdm(range(config['epochs'])):
        results = trainer.train()
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


pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                  'env_name': 'CentralControlPVInverterContinuousEnv',
                  'simulator': 'opendss',
                  'tracking_ids': ['inverter_s701a', 'adversary_inverter_s701a']}

create_env, env_name = make_create_env(pycigar_params, version=0)
register_env(env_name, create_env)

sim_params = input_parser('ieee37busdata')
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

    'num_workers': 3,
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
    "evaluation_interval": 5,
    "custom_eval_function": tune.function(custom_eval_function),
    'evaluation_config': {
        "seed": 42,
        # IMPORTANT NOTE: For policy gradients, this might not be the optimal policy
        'explore': False,
        'env_config': deepcopy(sim_params),
    },

    # ==== CUSTOM METRICS ====
    "callbacks": {
        "on_episode_start": tune.function(on_episode_start),
        "on_episode_step": tune.function(on_episode_step),
        "on_episode_end": tune.function(on_episode_end),
    },
}
# eval environment should not be random across workers
eval_start = 100  # random.randint(0, 3599 - 500)
base_config['evaluation_config']['env_config']['scenario_config']['start_end_time'] = [eval_start, eval_start + 750]
del base_config['evaluation_config']['env_config']['attack_randomization']
base_config['evaluation_config']['env_config']['scenario_config']['multi_config'] = False

if __name__ == '__main__':
    ray.init(local_mode=False)
    args = parse_cli_args()

    base_config['num_workers'] = args.workers
    base_config['evaluation_num_episodes'] = args.eval_rounds
    full_config = {
        'config': base_config,
        'epochs': args.epochs,
        'save_path': args.save_path,
        'algo': args.algo
    }

    if args.algo == 'ppo':
        
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

    elif args.algo == 'appo':
        config = deepcopy(full_config)
        run_hp_experiment(config, 'appo')

    ray.shutdown()