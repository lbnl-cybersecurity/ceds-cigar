import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from copy import deepcopy
import gym
from gym.envs.registration import register
import sys
from ray.tune import run_experiments


def make_create_env(params, version=0, render=None):
    exp_tag = params["exp_tag"]
    env_name = params["env_name"] + '-v{}'.format(version)
    def create_env(*_):
        sim_params = deepcopy(params['sim_params'])
        env_loc = 'pycigar.envs'
        try:
            register(
                id=env_name,
                entry_point=env_loc + ':{}'.format(params["env_name"]),
                kwargs={
                    "sim_params": sim_params,
                    "simulator": params['simulator']
                })
        except Exception:
            pass
        return gym.envs.make(env_name)

    return create_env, env_name

##########################################################################


import yaml
import os
import numpy as np
stream = open("sanity_check_pseudo_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

ray.init()
config = ppo.DEFAULT_CONFIG.copy()

config["num_gpus"] = 0
config["num_workers"] = 1
config["eager"] = False
# Call the utility function make_create_env to be able to
# register the Flow env for this experiment
pycigar_params = {"exp_tag": "exp_1",
                  "env_name": "PVInverterEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss"}

create_env, gym_name = make_create_env(params=pycigar_params, version=0)
register_env(gym_name, create_env)

trials = run_experiments({
    pycigar_params["exp_tag"]: {
        "run": "PPO",
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 1,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 999,
        "stop": {  # stopping conditions
            "training_iteration": 1,  # number of iterations to stop after
        },
    },
})

