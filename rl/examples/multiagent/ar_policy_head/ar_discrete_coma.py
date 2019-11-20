import argparse
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
from ar_discrete_policy_head import AutoregressiveActionsModel
from ar_discrete_policy_head import AutoregressiveOutput
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml

tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=0)


def fill_in_actions(info):
    """Callback that saves opponent actions into the agent obs.
    If you don't care about opponent actions you can leave this out."""
    to_update = info["post_batch"][SampleBatch.CUR_OBS]

    # set the opponent actions into the observation
    other_ids = list(info["all_pre_batches"].keys())
    other_ids.remove(info["agent_id"])
    action_flatten_dim = 5*len(other_ids)
    opponent_actions = None
    for other_id in other_ids:
        _, opponent_batch = info["all_pre_batches"][other_id]
        if opponent_actions is None:
            opponent_actions = opponent_batch[SampleBatch.ACTIONS]
        else:
            opponent_actions = np.concatenate((opponent_actions, opponent_batch[SampleBatch.ACTIONS]), axis=1)

    to_update[:, :action_flatten_dim] = opponent_actions


stream = open("../rl_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                  "env_name": "DiscreteMultiCOMAEnv",
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
    ModelCatalog.register_custom_model("autoregressive_model", AutoregressiveActionsModel)
    ModelCatalog.register_custom_action_dist("autoreg_output", AutoregressiveOutput)
    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config={
            # "eager": True,
            "env": env_name,
            "callbacks": {
                "on_postprocess_traj": fill_in_actions,
            },
            "lr": 0.0001,
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
        })
