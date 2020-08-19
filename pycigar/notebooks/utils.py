import argparse
import json
import math
import shutil
from collections import namedtuple
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from pycigar.utils.logging import logger
from pycigar.utils.output import plot_new
from pycigar.utils.registry import make_create_env
from ray.rllib.agents import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import _MultiAgentEnvToBaseEnv
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.evaluation.rollout_metrics import RolloutMetrics
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import merge_dicts
from ray.tune.registry import register_env


class EvalMetric(Enum):
    """Metrics to be maximized for the routine selecting the best policy over evaluation rounds"""
    REWARD_SUM = 1
    REWARD_MIN = 2
    REWARD_MAX = 3
    NEG_Y_U_SUM = 4
    NEG_Y_U_MAX = 5
    NEG_Y_U_MIN = 6


def get_metric_from_episodes(trainer: Trainer, episodes: List[RolloutMetrics], metric: EvalMetric) -> float:
    y_u = 'u' if trainer.global_vars.get('unbalance', False) else 'y'
    log_dicts = [ep.hist_data['logger']['log_dict'] for ep in episodes]

    # mean of [the sum of the y or u over the simulation] over all inverters
    sum_yu = [
        np.mean([
            sum(log_dict[k][y_u])
            for k in log_dict if y_u in log_dict[k]])
        for log_dict in log_dicts]

    if metric == EvalMetric.REWARD_SUM:
        return sum([ep.episode_reward for ep in episodes])
    elif metric == EvalMetric.REWARD_MIN:
        return min([ep.episode_reward for ep in episodes])
    elif metric == EvalMetric.REWARD_MAX:
        return max([ep.episode_reward for ep in episodes])
    elif metric == EvalMetric.NEG_Y_U_SUM:
        return -sum(sum_yu)
    elif metric == EvalMetric.NEG_Y_U_MAX:
        return -max(sum_yu)
    elif metric == EvalMetric.NEG_Y_U_MIN:
        return -min(sum_yu)
    else:
        raise NotImplementedError('Unsupported metric ' + metric)


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


    for i in range(len(episodes)):
        f = plot_new(episodes[i].hist_data['logger']['log_dict'], episodes[i].hist_data['logger']['custom_metrics'], trainer.iteration, trainer.global_vars['unbalance'])
        f.savefig(trainer.global_vars['reporter_dir'] + 'eval-epoch-' + str(trainer.iteration) + '_' + str(i+1) + '.png',
                bbox_inches='tight')
        plt.close(f) #OK

    for metric in EvalMetric:
        save_best_policy(trainer, episodes, metric)
    metrics = summarize_episodes(episodes)
    return metrics


def save_best_policy(trainer: Trainer, episodes: List[RolloutMetrics], metric: EvalMetric):
    """Save the best policy if the metric value is higher than for the last best policy"""
    metric_value = get_metric_from_episodes(trainer, episodes, metric)
    if 'best_eval' not in trainer.global_vars:
        trainer.global_vars['best_eval'] = {}

    if trainer.global_vars['best_eval'].get(metric.name, -np.Inf) < metric_value:
        trainer.global_vars['best_eval'][metric.name] = metric_value
        best_dir = Path(trainer.global_vars['reporter_dir']) / 'best' / metric.name.lower()
        best_dir.mkdir(exist_ok=True, parents=True)

        # save policy
        def save_policy_to_path(path):
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)

            # single policy
            if trainer.get_policy() is not None:
                trainer.get_policy().export_model(str(path))
            # multiple policies
            else:
                for k, p in trainer.optimizer.policies.items():
                    p_path = path / k
                    p.export_model(str(p_path))

        try:
            save_policy_to_path(best_dir / 'policy')
        except Exception:  # hack for lawrencium because it can't seem to fully delete the old dir
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

        # CSVs for the plots in the paper
        # only for one arbitrary episode
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
        info = {'epoch': trainer.iteration, 'hack_start': start, 'hack_end': end, 'metric': metric_value}
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


def get_base_config(env_name: str, cli_args: argparse.Namespace, sim_params: dict, config_update: dict = {}):
    """Factoring of the config code that was duplicated in every experiment file"""
    pycigar_params = {
        'exp_tag': 'experiment_ppo',
        'env_name': env_name,
        'simulator': 'opendss',
    }

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)

    # sometimes needed to init stuff, it really shouldn't be needed
    test_env = create_env(sim_params)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    base_config = {
        "env": env_name,
        "gamma": 0.5,
        'lr': 2e-3,
        'env_config': deepcopy(sim_params),
        'rollout_fragment_length': 35,
        'train_batch_size': max(500, 35 * cli_args.workers),
        'clip_param': 0.15,
        'lambda': 0.95,
        'vf_clip_param': 10000,
        'num_workers': cli_args.workers,
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
        'exploration_config': {'type': 'StochasticSampling', },  # default for PPO
        # ==== EVALUATION ====
        "evaluation_num_workers": 1, #0 if cli_args.local_mode else 4,
        'evaluation_num_episodes': 8, #cli_args.eval_rounds,
        "evaluation_interval": cli_args.eval_interval,
        "custom_eval_function": custom_eval_function,
        'evaluation_config': {
            "seed": 42,
            # IMPORTANT NOTE: For policy gradients, this might not be the optimal policy
            'explore': False,
            'env_config': deepcopy(sim_params),
        },
        # ==== CUSTOM METRICS ====
        "callbacks": CustomCallbacks,
    }
    base_config['env_config']['attack_randomization']['generator'] = 'UnbalancedAttackDefinitionGeneratorEvaluationRandom'
    base_config['evaluation_config']['env_config']['attack_randomization']['generator'] = 'UnbalancedAttackDefinitionGeneratorEvaluation'

    config = merge_dicts(base_config, config_update)
    return config, create_env
