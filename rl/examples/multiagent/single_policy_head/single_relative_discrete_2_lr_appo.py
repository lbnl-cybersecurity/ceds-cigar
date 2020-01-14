import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import APPOTrainer
import argparse
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml


SAVE_RATE = 5

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
                  "env_name": "SingleRelativeDiscreteCoopEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ['pv_8', 'pv_17', 'pv_12']}
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
def coop_train_full(config, reporter):

    # initialize PPO agent on environment. This may create 1 or more workers.
    agent1 = APPOTrainer(env=env_name, config=config)

    # begin train iteration
    for i in range(5000):
        # this function will collect samples and train agent with the samples.
        # result is the summerization of training progress.
        result = agent1.train()

        # adding a phase into the result dictionary.
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]

        # for every SAVE_RATE training iterations, we test the agent on the test environment to see how it performs.
        if i != 0 and (i+1) % SAVE_RATE == 0:
            state = agent1.save('~/ray_results/checkpoint')
            done = False
            # reset the test environment
            obs = test_env.reset()
            while not done:
                act = {}
                for k, v in obs.items():
                    # for each observation, let the policy decides what to do
                    act[k] = agent1.compute_action(v, policy_id='pol')
                # forward 1 step with agent action
                obs, _, done, _ = test_env.step(act)
                done = done['__all__']
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
        'lr': 5e-03,
        'sample_batch_size': 500,
        'lambda': 0.99,
        'train_batch_size': 200,
        #'batch_mode': 'complete_episodes',
        #'lr_schedule': [[0, 5e-03], [400000, 5e-03], [400001, 5e-04]],
        #worker
        'num_workers': 5,
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
        #'use_kl_loss': True,
        #"num_sgd_iter": 30,
        "minibatch_buffer_size": 128,
        'vf_loss_coeff': 0.7, 'entropy_coeff': 0.0001,
        'vtrace_clip_rho_threshold': 1000, 'vtrace_clip_pg_rho_threshold': 1000,
        #"grad_clip": 10.0,
        #model
        'model': {'conv_filters': None, 'conv_activation': 'tanh',
                  'fcnet_activation': 'tanh', 'fcnet_hiddens': [256, 128, 64, 32],
                  'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True,
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
    tune.run(coop_train_full, config=config)
