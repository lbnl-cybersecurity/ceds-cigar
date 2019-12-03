import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import argparse
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml

SAVE_RATE = 10

"""
Parser to pass argument from terminal command
--run: RL algorithm, ex. PG, PPO, IMPALA
--stop: stop criteria of experiment. The experiment will stop when mean reward reach to this value.
Example of terminal command:
  > python single_relative_discrete_2_lr.py --run PPO --stop 0
"""
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=0)

"""
Load the scenarios configuration file. This file contains the scenario information
for the experiment.
"""
stream = open("../rl_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

"""
Register the environment to OpenGymAI. This is necessary, RLlib can find the new environment
with string name env_name_v:version:, ex. SingleRelativeDiscreteCoopEnv_v0.
env_name: name of environment being used.
sim_params: simulation params, it is the scenario configuration.
simulator: the simulator being used, ex. opendss, gridlabd...
tracking_ids: list of ids of devices being tracked during the experiment.
"""
pycigar_params1 = {"exp_tag": "cooperative_multiagent_ppo",
                   "env_name": "SingleDiscreteCoopEnv",
                   "sim_params": sim_params,
                   "simulator": "opendss",
                   "tracking_ids": ['pv_8', 'pv_17', 'pv_12']}

"""
call function make_create_env() to register the new environment to OpenGymAI.
create_env() is a function to create new instance of the environment.
env_name: the registered name of the new environment.
"""
create_env1, env_name1 = make_create_env(params=pycigar_params1, version=0)
register_env(env_name1, create_env1)

test_env1 = create_env1()
obs_space = test_env1.observation_space  # get the observation space, we need this to construct our agent(s) observation input
act_space = test_env1.action_space  # get the action space, we need this to construct our agent(s) action output

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

"""
Define training process. Ray/Tune will call this function with config params.
"""
def coop_train_fn(config, reporter):

    # initialize PPO agent on 1st environment. This may create 1 or more workers.
    agent1 = PPOTrainer(env=env_name1, config=config)

    # begin train iteration
    for i in range(50):
        # this function will collect samples and train agent with the samples.
        # result is the summerization of training progress.
        result = agent1.train()

        # adding a phase into the result dictionary.
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]

        # for every SAVE_RATE training iterations, we test the agent on the test environment to see how it performs.
        if i != 0 and (i+1) % SAVE_RATE == 0:
            done = False
            # reset the test environment
            obs = test_env1.reset()
            while not done:
                act = {}
                for k, v in obs.items():
                    # for each observation, let the policy decides what to do
                    act[k] = agent1.compute_action(v, policy_id='pol')
                # forward 1 step with agent action
                obs, _, done, _ = test_env1.step(act)
                done = done['__all__']
            # plot the result. This will be saved in ./results
            test_env1.plot(pycigar_params1['exp_tag'], env_name1, i+1)
    # save the params of agent
    state = agent1.save()
    # stop the agent
    agent1.stop()

    # create new agent, over 2nd environment, new config
    agent2 = PPOTrainer(env=env_name2, config=config)
    # restore the current state (params of nn, etc...) of agent1 on agent2
    agent2.restore(state)
    for i in range(50):
        result = agent2.train()
        result["phase"] = 2
        result["timesteps_total"] += phase1_time  # keep timestamp moving forward
        phase2_time = result["timesteps_total"]
        reporter(**result)
        if i != 0 and (i+1) % SAVE_RATE == 0:
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
    #config for RLlib - PPO agent
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
            # list of all policies
            "policies": {
                "pol": (None, obs_space, act_space, {}),
            },
            # the mapping function between agents and policies. We map all agents to 1 policy.
            "policy_mapping_fn": lambda x: "pol",
        }
    }
    # call tune.run() to run the coop_train_fn() with the config() above
    tune.run(coop_train_fn, config=config)
