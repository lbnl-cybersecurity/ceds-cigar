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

pycigar_params1 = {"exp_tag": "cooperative_multiagent_ppo",
                   "env_name": "SingleDiscreteCoopEnv",
                   "sim_params": sim_params,
                   "simulator": "opendss",
                   "tracking_ids": ['pv_8', 'pv_17', 'pv_12']}


create_env1, env_name1 = make_create_env(params=pycigar_params1, version=0)
register_env(env_name1, create_env1)

test_env1 = create_env1()
obs_space = test_env1.observation_space
act_space = test_env1.action_space

pycigar_params2 = {"exp_tag": "cooperative_multiagent_ppo",
                   "env_name": "SecondStageSingleDiscreteCoopEnv",
                   "sim_params": sim_params,
                   "simulator": "opendss",
                   "tracking_ids": ['pv_8', 'pv_17', 'pv_12']}


create_env2, env_name2 = make_create_env(params=pycigar_params2, version=0)
register_env(env_name2, create_env2)

test_env2 = create_env2()
obs_space = test_env2.observation_space
act_space = test_env2.action_space


def coop_train_fn(config, reporter):
    agent1 = PPOTrainer(env=env_name1, config=config)
    for i in range(50):
        result = agent1.train()
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]
        if i != 0 and (i+1) % 10 == 0:
            done = False
            obs = test_env1.reset()
            while not done:
                act = {}
                for k, v in obs.items():
                    act[k] = agent1.compute_action(v, policy_id='pol')
                obs, _, done, _ = test_env1.step(act)
                done = done['__all__']
            test_env1.plot(pycigar_params1['exp_tag'], env_name1, i+1)
    state = agent1.save()
    agent1.stop()

    agent2 = PPOTrainer(env=env_name2, config=config)
    agent2.restore(state)
    for i in range(50):
        result = agent2.train()
        result["phase"] = 2
        result["timesteps_total"] += phase1_time  # keep time moving forward
        phase2_time = result["timesteps_total"]
        reporter(**result)
        if (i+1) % 10 == 0:
            done = False
            obs = test_env2.reset()
            while not done:
                act = {}
                for k, v in obs.items():
                    act[k] = agent2.compute_action(v, policy_id='pol')
                obs, _, done, _ = test_env2.step(act)
                done = done['__all__']
            test_env2.plot(pycigar_params2['exp_tag'], env_name2, i+1)
    state = agent2.save()
    agent2.stop()


if __name__ == "__main__":
    ray.init()
    config = {
        "gamma": 0.5,
        'lr': 5e-05,
        'sample_batch_size': 50,
        "vf_clip_param": 500.0,
        'entropy_coeff_schedule': [[0, 0], [150000, 0.000000000001]],
        'model': {'conv_filters': None, 'conv_activation': 'tanh',
                  'fcnet_activation': 'tanh', 'fcnet_hiddens': [512, 256, 128, 64, 32],
                  'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False,
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
