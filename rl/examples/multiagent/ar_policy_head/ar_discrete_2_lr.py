import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import argparse
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml
from ar_discrete_policy_head import AutoregressiveActionsModel
from ar_discrete_policy_head import AutoregressiveOutput

tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=0)


ModelCatalog.register_custom_model("autoregressive_model", AutoregressiveActionsModel)
ModelCatalog.register_custom_action_dist("autoreg_output", AutoregressiveOutput)

stream = open("../rl_config_scenarios.yaml", "r")
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
    for i in range(100):
        agent1 = PPOTrainer(env=env_name, config=config)
        for i in range(100):
            result = agent1.train()
            result["phase"] = 1
            reporter(**result)
            phase1_time = result["timesteps_total"]
            if i % 10 == 0:
                done = False
                obs = test_env.reset()
                while not done:
                    act = {}
                    for k, v in obs.items():
                        act[k] = agent1.compute_action(v, policy_id='pol')
                    obs, _, done, _ = test_env.step(act)
                    done = done['__all__']
                test_env.plot()
        agent1.stop()


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
        "lr": 5e-5,
        'sample_batch_size': 50,
        "train_batch_size": 1000,
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
