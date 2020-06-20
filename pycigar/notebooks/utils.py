import argparse
import json
import math
import os
import shutil
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from pycigar.utils.logging import logger
from pycigar.utils.output import plot_new
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


def custom_eval_function(trainer, eval_workers):
    if trainer.config["evaluation_num_workers"] == 0:
        for _ in range(trainer.config["evaluation_num_episodes"]):
            eval_workers.local_worker().sample()

    else:
        num_rounds = int(
            math.ceil(trainer.config["evaluation_num_episodes"] / trainer.config["evaluation_num_workers"])
        )
        for i in range(num_rounds):
            ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    episodes, _ = collect_episodes(eval_workers.local_worker(), eval_workers.remote_workers())
    metrics = summarize_episodes(episodes)

    f = plot_new(
        episodes[-1].hist_data['logger']['log_dict'],
        episodes[-1].hist_data['logger']['custom_metrics'],
        trainer.iteration,
        trainer.global_vars['unbalance'],
    )
    f.savefig(
        trainer.global_vars['reporter_dir'] + 'eval-epoch-' + str(trainer.iteration) + '.png', bbox_inches='tight'
    )
    plt.close(f)

    save_best_policy(trainer, episodes)
    return metrics


def save_best_policy(trainer, episodes):
    mean_r = np.array([ep.episode_reward for ep in episodes]).mean()
    if 'best_eval_reward' not in trainer.global_vars or trainer.global_vars['best_eval_reward'] < mean_r:
        os.makedirs(os.path.join(trainer.global_vars['reporter_dir'], 'best'), exist_ok=True)
        trainer.global_vars['best_eval_reward'] = mean_r
        # save policy
        policy_path = os.path.join(trainer.global_vars['reporter_dir'], 'best', 'policy')
        if os.path.exists(policy_path):
            shutil.rmtree(policy_path, ignore_errors=True)

        trainer.get_policy().export_model(policy_path + '_' + str(trainer.iteration))
        # save plots
        ep = episodes[-1]
        data = ep.hist_data['logger']['log_dict']
        f = plot_new(
            data, ep.hist_data['logger']['custom_metrics'], trainer.iteration, trainer.global_vars['unbalance']
        )
        f.savefig(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'eval.png'))
        plt.close(f)
        # save CSV
        k = [k for k in data if k.startswith('inverter_s701')][0]

        ep_hist = pd.DataFrame(
            dict(v=data[data[k]['node']]['voltage'], y=data[k]['y'], q_set=data[k]['q_set'], q_val=data[k]['q_out'])
        )
        a_hist = pd.DataFrame(data[k]['control_setting'], columns=['a1', 'a2', 'a3', 'a4', 'a5'])
        adv_a_hist = pd.DataFrame(
            data['adversary_' + k]['control_setting'], columns=['adv_a1', 'adv_a2', 'adv_a3', 'adv_a4', 'adv_a5']
        )
        translation, slope = get_translation_and_slope(data[k]['control_setting'])
        adv_translation, adv_slope = get_translation_and_slope(data['adversary_' + k]['control_setting'])
        trans_slope_hist = pd.DataFrame(
            dict(translation=translation, slope=slope, adv_translation=adv_translation, adv_slope=adv_slope)
        )

        df = ep_hist.join(a_hist, how='outer')
        df = df.join(adv_a_hist, how='outer')
        df = df.join(trans_slope_hist, how='outer')
        df.to_csv(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'last_eval_hists.csv'))

        # save info
        start = ep.custom_metrics["hack_start"]
        end = ep.custom_metrics["hack_end"]
        info = {'epoch': trainer.iteration, 'hack_start': start, 'hack_end': end, 'reward': mean_r}
        with open(os.path.join(trainer.global_vars['reporter_dir'], 'best', 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)


def get_translation_and_slope(a_val):
    points = np.array(a_val)
    slope = points[:, 1] - points[:, 0]
    og_point = points[0, 2]
    translation = points[:, 2] - og_point
    return translation, slope


def get_custom_callbacks():
    ActionTuple = namedtuple('Action', ['action', 'timestep'])

    def on_episode_start(info):
        episode = info["episode"]
        episode.user_data["num_actions_taken"] = 0
        episode.user_data["magnitudes"] = []
        episode.user_data["true_actions"] = [ActionTuple(2, -1)]  # 2 is the init action
        episode.hist_data["y"] = []

        # get the base env
        env = info['env'].get_unwrapped()[0]
        episode.user_data["tracking_id"] = env.k.device.get_rl_device_ids()[0]

    def on_episode_step(info):
        episode = info["episode"]
        action = episode.last_action_for()
        if (action != episode.user_data["true_actions"][-1].action).all():
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

        tracking = logger()
        info['episode'].hist_data['logger'] = {'log_dict': tracking.log_dict, 'custom_metrics': tracking.custom_metrics}

        env = info['env'].vector_env.envs[0]
        t_id = env.k.device.get_rl_device_ids()[0]
        hack_start = int([k for k, v in env.k.scenario.hack_start_times.items() if 'adversary_' + t_id in v][0])
        hack_end = int([k for k, v in env.k.scenario.hack_end_times.items() if 'adversary_' + t_id in v][0])
        episode.custom_metrics["hack_start"] = hack_start
        episode.custom_metrics["hack_end"] = hack_end

    return {
        "on_episode_start": on_episode_start,
        "on_episode_step": on_episode_step,
        "on_episode_end": on_episode_end,
    }


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs per trial')
    parser.add_argument('--save-path', type=str, default='~/hp_experiment3', help='where to save the results')
    parser.add_argument('--workers', type=int, default=3, help='number of cpu workers per run')
    parser.add_argument(
        '--eval-rounds', type=int, default=1, help='number of evaluation rounds to run to smooth random results'
    )
    parser.add_argument('--eval-interval', type=int, default=5, help='do an evaluation every N epochs')
    parser.add_argument(
        "--algo", help="use PPO or APPO", choices=['ppo', 'appo'], nargs='?', const='ppo', default='ppo', type=str.lower
    )
    parser.add_argument('--local-mode', action='store_true')
    parser.add_argument('--redis-pwd', type=str)
