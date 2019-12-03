from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml
import argparse
import ray
from ray import tune

"""
Different way to call ray.run()
"""

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=0)

stream = open("../rl_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                  "env_name": "SingleDiscreteCoopEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ['pv_8', 'pv_9', 'pv_12']}


create_env, env_name = make_create_env(params=pycigar_params, version=0)
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config={
            "env": env_name,
            "gamma": 0.5,
            'lr': 5e-04,
            'model': {'conv_filters': None, 'conv_activation': 'tanh',
                      'fcnet_activation': 'tanh', 'fcnet_hiddens': [256, 128, 64, 32],
                      'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True,
                      'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256,
                      'lstm_use_prev_action_reward': False, 'state_shape': None,
                      'framestack': False, 'dim': 84, 'grayscale': False,
                      'zero_mean': True, 'custom_preprocessor': None,
                      'custom_model': None, 'custom_action_dist': None, 'custom_options': {}},
            "num_gpus": 0,
            'multiagent': {
                "policies": {
                    "pol": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda x: "pol",
            }
        })
