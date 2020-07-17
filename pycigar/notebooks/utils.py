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
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import _MultiAgentEnvToBaseEnv
from pathlib import Path

def custom_eval_function(trainer, eval_workers):
    def set_hack(env, hack):
        for node in env.k.sim_params['scenario_config']['nodes']:
            for d in node['devices']:
                d['hack'][1] = hack

    if trainer.config["evaluation_num_workers"] == 0:
        hacks = [0.45] * trainer.config["evaluation_num_episodes"]
        eval_workers.local_worker().foreach_env(lambda env: set_hack(env, hacks[0]))

        for _ in range(trainer.config["evaluation_num_episodes"]):
            eval_workers.local_worker().sample()
    else:
        num_rounds = int(
            math.ceil(trainer.config["evaluation_num_episodes"] / trainer.config["evaluation_num_workers"])
        )
        workers = eval_workers.remote_workers()
        hacks = np.linspace(0.1, 0.5, trainer.config["evaluation_num_workers"]).tolist() * trainer.config["evaluation_num_episodes"]

        for w, h in zip(workers, hacks):
            w.foreach_env.remote(lambda env: set_hack(env, h))

        for i in range(num_rounds):
            ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    episodes, _ = collect_episodes(eval_workers.local_worker(), eval_workers.remote_workers())
    metrics = summarize_episodes(episodes)

    for ep, hack in zip(episodes[:max(1, trainer.config["evaluation_num_workers"])+1], hacks):
        dir = Path(trainer.global_vars['reporter_dir']) / f'hack_{hack}'
        dir.mkdir(exist_ok=True, parents=True)
        f = plot_new(
            ep.hist_data['logger']['log_dict'],
            ep.hist_data['logger']['custom_metrics'],
            trainer.iteration,
            trainer.global_vars.get('unbalance', False),
            trainer.global_vars.get('multiagent', False),
        )
        f.savefig(str(dir / f'eval-epoch-{trainer.iteration}-hack-{hack:.2f}.png'),
                bbox_inches='tight')
        plt.close(f)

    for m in ["reward_sum", "reward_min", "neg_y_u_sum", "neg_y_u_max"]:
        save_best_policy(trainer, episodes, m)
    return metrics


def get_metric_from_episodes(trainer, episodes, metric):
    assert metric in ["reward_sum", "reward_min", "neg_y_u_sum", "neg_y_u_max"]

    y_u = 'u' if trainer.global_vars.get('unbalance', False) else 'y'
    log_dicts = [ep.hist_data['logger']['log_dict'] for ep in episodes]
    sum_us = [np.mean([sum(log_dict[k][y_u]) for k in log_dict if y_u in log_dict[k]]) for log_dict in log_dicts]

    if metric == "reward_sum":
        return np.array([ep.episode_reward for ep in episodes]).sum()
    elif metric == "reward_min":
        return np.array([ep.episode_reward for ep in episodes]).min()
    elif metric == "neg_y_u_sum":
        return sum(sum_us)
    elif metric == "neg_y_u_max":
        return max(sum_us)


def save_best_policy(trainer, episodes, metric_name):
    metric = get_metric_from_episodes(trainer, episodes, metric_name)
    if 'best_eval' not in trainer.global_vars:
        trainer.global_vars['best_eval'] = {}

    if trainer.global_vars['best_eval'].get(metric_name, -np.Inf) < metric:
        trainer.global_vars['best_eval'][metric_name] = metric
        best_dir = Path(trainer.global_vars['reporter_dir']) / 'best' / metric_name
        best_dir.mkdir(exist_ok=True, parents=True)

        # save policy
        def save_policy_to_path(path):
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)

            if trainer.get_policy() is not None:
                trainer.get_policy().export_model(str(path))
            else:
                for k, p in trainer.optimizer.policies.items():
                    p_path = path / k
                    p.export_model(str(p_path))

        try:
            save_policy_to_path(best_dir / 'policy')
        except Exception: # hack for lawrencium
            save_policy_to_path(best_dir / f'policy_{trainer.iteration}')

        # save plots
        for i, ep in enumerate(episodes):
            data = ep.hist_data['logger']['log_dict']
            f = plot_new(
                data, ep.hist_data['logger']['custom_metrics'], trainer.iteration,
                trainer.global_vars.get('unbalance', False),
                trainer.global_vars.get('multiagent', False),
            )
            f.savefig(str(best_dir / f'eval_{i}.png'))
            plt.close(f)

        ep = episodes[-1]
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
        df.to_csv(str(best_dir / 'last_eval_hists.csv'))

        # save info
        start = ep.custom_metrics["hack_start"]
        end = ep.custom_metrics["hack_end"]
        info = {'epoch': trainer.iteration, 'hack_start': start, 'hack_end': end, 'metric': metric}
        with open(str(best_dir / 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)


def get_translation_and_slope(a_val):
    points = np.array(a_val)
    slope = points[:, 1] - points[:, 0]
    og_point = points[0, 2]
    translation = points[:, 2] - og_point
    return translation, slope


class CustomCallbacks(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict=None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        self.ActionTuple = namedtuple('Action', ['action', 'timestep'])

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data["num_actions_taken"] = 0
        episode.user_data["magnitudes"] = []
        episode.user_data["true_actions"] = [self.ActionTuple(2, -1)]  # 2 is the init action
        episode.hist_data["y"] = []

        # get the base env
        env = base_env.get_unwrapped()[0]
        episode.user_data["tracking_id"] = env.k.device.get_rl_device_ids()[0]

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        action = episode.last_action_for()
        if (action != episode.user_data["true_actions"][-1].action).all():
            episode.user_data["true_actions"].append(self.ActionTuple(action, episode.length))

        if episode.last_info_for() is not None:
            y = episode.last_info_for()[episode.user_data["tracking_id"]]['y']
            episode.hist_data["y"].append(y)

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        actions = episode.user_data["true_actions"]
        avg_mag = (np.array([t.action for t in actions[1:]]) - 2).mean()
        num_actions = len(actions) - 1
        if num_actions > 0:
            episode.custom_metrics['latest_action'] = actions[-1].timestep
            episode.custom_metrics['earliest_action'] = actions[1].timestep

        episode.custom_metrics["avg_magnitude"] = avg_mag
        episode.custom_metrics["num_actions_taken"] = num_actions

        tracking = logger()
        episode.hist_data['logger'] = {'log_dict': tracking.log_dict, 'custom_metrics': tracking.custom_metrics}

        if isinstance(base_env, _MultiAgentEnvToBaseEnv):
            env = base_env.envs[0]
        else:
            env = base_env.vector_env.envs[0]
        t_id = env.k.device.get_rl_device_ids()[0]
        hack_start = int([k for k, v in env.k.scenario.hack_start_times.items() if 'adversary_' + t_id in v][0])
        hack_end = int([k for k, v in env.k.scenario.hack_end_times.items() if 'adversary_' + t_id in v][0])
        episode.custom_metrics["hack_start"] = hack_start
        episode.custom_metrics["hack_end"] = hack_end


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs per trial')
    parser.add_argument('--save-path', type=str, default='~/hp_experiment3', help='where to save the results')
    parser.add_argument('--workers', type=int, default=3, help='number of cpu workers per run')
    parser.add_argument(
        '--eval-rounds', type=int, default=1, help='number of evaluation rounds to run to smooth random results'
    )
    parser.add_argument('--eval-interval', type=int, default=1, help='do an evaluation every N epochs')
    parser.add_argument(
        "--algo", help="use PPO or APPO", choices=['ppo', 'appo'], nargs='?', const='ppo', default='ppo', type=str.lower
    )
    parser.add_argument('--local-mode', action='store_true')
    parser.add_argument('--redis-pwd', type=str)
