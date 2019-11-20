import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import argparse
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml
import pycigar.config as config_pycigar
from ar_discrete_policy_head import AutoregressiveActionsModel
from ar_discrete_policy_head import AutoregressiveOutput
import os
import random

tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=0)


ModelCatalog.register_custom_model("autoregressive_model", AutoregressiveActionsModel)
ModelCatalog.register_custom_action_dist("autoreg_output", AutoregressiveOutput)

stream = open("../multiagent/rl_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                  "env_name": "ARDiscreteCoopEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ['pv_8', 'pv_17', 'pv_12']}


create_env, env_name = make_create_env(params=pycigar_params, version=0)
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def coop_train_fn(config, reporter):

    phase = 0
    state = None

    for _ in range(1000):

        # form new pycigar_params
        filenames = [x for x in os.listdir(os.path.join(config_pycigar.LOG_DIR, 'bad_scenarios/train')) if x.endswith(".yml")]
        filename = random.choice(filenames)
        filename = os.path.join(config_pycigar.LOG_DIR, 'bad_scenarios/train/{}'.format(filename))
        stream = open(filename, "r")
        sim_params = yaml.safe_load(stream)

        pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                          "env_name": "ARDiscreteCoopEnv",
                          "sim_params": sim_params,
                          "simulator": "opendss",
                          "tracking_ids": ['pv_8', 'pv_9', 'pv_12']}

        pycigar_params['sim_params']['scenario_config']['nodes'][8]['devices'][0]['controller'] = 'rl_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][9]['devices'][0]['controller'] = 'rl_controller'
        create_env, env_name = make_create_env(params=pycigar_params, version=0)
        register_env(env_name, create_env)

        agent = PPOTrainer(env=env_name, config=config)
        if phase != 0:
            agent.restore(state)
        for _ in range(3):
            result = agent.train()
            result["phase"] = phase
            reporter(**result)
        phase += 1
        state = agent.save()
        agent.stop()


if __name__ == "__main__":
    ray.init()
    config = {
        # Size of batches collected from each worker
        #"sample_batch_size": 400,
        # Number of timesteps collected for each SGD round
        #"train_batch_size": 1200,
        #Total SGD batch size across all devices for SGD
        #sgd_minibatch_size": 512,
        # "vf_clip_param": 500.0,
        # "lr": 5e-5,
        "gamma": 0.5,
        "num_gpus": 0,
        "model": {
            "custom_model": "autoregressive_model",
            "custom_action_dist": "autoreg_output",
        },
        'multiagent': {
            "policies": {
                "pol": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda x: "pol",
        }
    }
    tune.run(coop_train_fn, config=config)
