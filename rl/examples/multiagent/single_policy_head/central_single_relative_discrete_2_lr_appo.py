import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import APPOTrainer
import argparse
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml
import time

SAVE_RATE = 2

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

pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                  "env_name": "CentralControlPVInverterEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ['pv_17']}
"""
call function make_create_env() to register the new environment to OpenGymAI.
create_env() is a function to create new instance of the environment.
env_name: the registered name of the new environment.
"""
create_env, env_name = make_create_env(params=pycigar_params, version=0)
register_env(env_name, create_env)
test_env = create_env()
obs_space = test_env.observation_space  # get the observation space, we need this to construct our agent(s) observation input
act_space = test_env.action_space  # get the action space, we need this to construct our agent(s) action output


"""
Define training process. Ray/Tune will call this function with config params.
"""
def coop_train_fn(config, reporter):

    # initialize PPO agent on environment. This may create 1 or more workers.
    agent1 = APPOTrainer(env=env_name, config=config)

    # begin train iteration
    for i in range(100):
        # this function will collect samples and train agent with the samples.
        # result is the summerization of training progress.
        result = agent1.train()
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]

        # for every SAVE_RATE training iterations, we test the agent on the test environment to see how it performs.
        if i != 0 and (i+1) % SAVE_RATE == 0:
            #agent1.get_policy().export_model('/Users/toanngo/policy/' + str(i+1))
            state = agent1.save('~/ray_results/checkpoint')
            done = False
            start_time = time.time()
            obs = test_env.reset()
            while not done:
                # for each observation, let the policy decides what to do
                act = agent1.compute_action(obs)
                # forward 1 step with agent action
                obs, _, done, _ = test_env.step(act)
            end_time = time.time()
            ep_time = end_time-start_time
            print("\n Episode time is ")
            print(ep_time)
            print("\n")
            # plot the result. This will be saved in ./results
            test_env.plot(pycigar_params['exp_tag'], env_name, i+1)
    # save the params of agent
    # state = agent1.save()
    # stop the agent
    agent1.stop()


if __name__ == "__main__":
    ray.init()
    #config for RLlib - PPO agent
    config = {
        'vtrace': True,
        "gamma": 0.99,
        'lr': 5e-04,
        'lambda': 0.99,
        'sample_batch_size': 50,
        'train_batch_size': 256,
        #'lr_schedule': [[0, 5e-04], [12000, 5e-04], [13500, 5e-05]],
        #worker
        'num_workers': 3,
        'num_gpus': 0,
        'num_cpus_per_worker': 1,
        'num_gpus_per_worker': 0,
        'custom_resources_per_worker': {},
        'num_cpus_for_driver': 1,
        'memory': 0,
        'object_store_memory': 0,
        'memory_per_worker': 0,
        'object_store_memory_per_worker': 0,
        'num_envs_per_worker': 1,
        'evaluation_num_episodes': 1,
        #model
        'model': {'conv_filters': None, 'conv_activation': 'tanh',
                  'fcnet_activation': 'tanh', 'fcnet_hiddens': [128, 64, 32],
                  'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True,
                  'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256,
                  'lstm_use_prev_action_reward': False, 'state_shape': None,
                  'framestack': False, 'dim': 84, 'grayscale': False,
                  'zero_mean': True, 'custom_preprocessor': None,
                  'custom_model': None, 'custom_action_dist': None, 'custom_options': {}},
    }

    # call tune.run() to run the coop_train_fn() with the config() above
    tune.run(coop_train_fn, config=config)
