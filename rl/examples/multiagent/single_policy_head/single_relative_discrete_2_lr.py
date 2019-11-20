import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import argparse
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml


tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=0)

stream = open("../rl_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                  "env_name": "SingleRelativeDiscreteCoopEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ['pv_8', 'pv_17', 'pv_12']}


create_env, env_name = make_create_env(params=pycigar_params, version=0)
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def coop_train_fn(config, reporter):
    agent1 = PPOTrainer(env=env_name, config=config)
    for i in range(100):
        result = agent1.train()
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]
        if i != 0 and (i+1) % 10 == 0:
            done = False
            obs = test_env.reset()
            while not done:
                act = {}
                for k, v in obs.items():
                    act[k] = agent1.compute_action(v, policy_id='pol')
                obs, _, done, _ = test_env.step(act)
                done = done['__all__']
            test_env.plot(pycigar_params['exp_tag'], env_name, i)
    state = agent1.save()
    agent1.stop()


if __name__ == "__main__":
    ray.init()
    config = {
        "gamma": 0.5,
        'lr': 5e-05,
        'sample_batch_size': 50,
        'model': {'conv_filters': None, 'conv_activation': 'tanh',
                  'fcnet_activation': 'tanh', 'fcnet_hiddens': [256, 128, 64, 32],
                  'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True,
                  'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256,
                  'lstm_use_prev_action_reward': False, 'state_shape': None,
                  'framestack': False, 'dim': 84, 'grayscale': False,
                  'zero_mean': True, 'custom_preprocessor': None,
                  'custom_model': None, 'custom_action_dist': None, 'custom_options': {}},
        'multiagent': {
            "policies": {
                "pol": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda x: "pol",
        }
    }
    tune.run(coop_train_fn, config=config)
