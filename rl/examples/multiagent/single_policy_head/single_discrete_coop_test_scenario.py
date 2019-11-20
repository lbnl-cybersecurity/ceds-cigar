import argparse
import ray
from ray import tune
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml
import os
import pycigar.config as config_pycigar


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=0)

filename = 'data_f740b21e1a124776848c416945fa6409.yml'
filename = os.path.join(config_pycigar.LOG_DIR, 'bad_scenarios/train/{}'.format(filename))
stream = open(filename, "r")
sim_params = yaml.safe_load(stream)
pycigar_params = {"env_name": "SingleDiscreteCoopEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ["pv_8", "pv_9"]}
pycigar_params['sim_params']['scenario_config']['nodes'][8]['devices'][0]['controller'] = 'rl_controller'
pycigar_params['sim_params']['scenario_config']['nodes'][9]['devices'][0]['controller'] = 'rl_controller'


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
            "num_gpus": 0,
            'multiagent': {
                "policies": {
                    "pol": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda x: "pol",
            }
        })

