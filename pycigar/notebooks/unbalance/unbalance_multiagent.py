import argparse
import os
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pycigar
import ray
from pycigar.notebooks.utils import add_common_args, get_base_config
from pycigar.utils.input_parser import input_parser
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.policy.policy import Policy
from tqdm import tqdm

from pycigar.notebooks.unbalance.unbalance import set_unbalance_attack, adjust_default_curves


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Experimentations of the unbalance attack for multi-agent')
    parser.add_argument('--continuous', action='store_true')
    add_common_args(parser)
    return parser.parse_args()


def run_train(config, reporter):
    trainer = PPOTrainer(config=config['config'])

    # needed so that the custom eval fn knows where to save plots
    trainer.global_vars['reporter_dir'] = reporter.logdir
    trainer.global_vars['unbalance'] = True  # for plots
    trainer.global_vars['multiagent'] = True  # for plots

    for _ in tqdm(range(config['epochs'])):
        results = trainer.train()
        #        del results['hist_stats']['logger']  # don't send to tensorboard
        #       if 'evaluation' in results:
        #          del results['evaluation']['hist_stats']['logger']
        reporter(**results)

    trainer.stop()


def run_experiment(full_config, name):
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


def policy_mapping_fn(agent_id):
    if agent_id.endswith('a'):
        return "phase_a"
    elif agent_id.endswith('b'):
        return "phase_b"
    elif agent_id.endswith('c'):
        return "phase_c"

    return "do_nothing"


class DoNothingPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [0 for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


if __name__ == '__main__':
    args = parse_cli_args()
    feeder_path = Path(pycigar.DATA_DIR) / 'ieee37busdata_regulator_attack'

    sim_params = input_parser(misc_inputs_path=str(feeder_path / 'misc_inputs.csv'),
                              dss_path=str(feeder_path / 'ieee37.dss'),
                              load_solar_path=str(feeder_path / 'load_solar_data.csv'),
                              breakpoints_path=str(feeder_path / 'breakpoints.csv'))
    base_config, create_env = \
        get_base_config(env_name='PhaseSpecificContinuousMultiEnv',
                        cli_args=args,
                        sim_params=sim_params)

    set_unbalance_attack(base_config)
    adjust_default_curves(base_config)

    test_env = create_env(sim_params)
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    model = {
        'fcnet_activation': 'tanh',
        'fcnet_hiddens': [64, 64, 32],
        'free_log_std': False,
        'vf_share_layers': True,
        'use_lstm': False,
        'state_shape': None,
        'framestack': False,
        'zero_mean': True,
    }
    base_config['multiagent'] = {
        "policies_to_train": ["phase_a", "phase_b", "phase_c"],
        "policies": {
            "phase_a": (PPOTFPolicy, obs_space, act_space, model),
            "phase_b": (PPOTFPolicy, obs_space, act_space, model),
            "phase_c": (PPOTFPolicy, obs_space, act_space, model),
            "do_nothing": (DoNothingPolicy, obs_space, act_space, {})
        },
        "policy_mapping_fn": policy_mapping_fn
    }

    if args.redis_pwd:
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
    config['config']['env_config']['M'] = 1000
    config['config']['env_config']['N'] = 50
    config['config']['env_config']['P'] = 100
    config['config']['lr'] = 1e-3
    config['config']['clip_param'] = 0.2

    run_experiment(config, 'main')

    ray.shutdown()
