import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import APPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
import random
from copy import deepcopy
import argparse
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml
import time
from pycigar.utils.input_parser import input_parser

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
sim_params = input_parser('ieee37busdata')

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
                  "simulator": "opendss",
                  "tracking_ids": ['inverter_s701a', 'creg1a', 'creg1c']}
"""
call function make_create_env() to register the new environment to OpenGymAI.
create_env() is a function to create new instance of the environment.
env_name: the registered name of the new environment.
"""
create_env, env_name = make_create_env(pycigar_params, version=0)
register_env(env_name, create_env)
eval_start = random.randint(0, 3599-500)
eval_params = deepcopy(sim_params)   
eval_params['scenario_config']['start_end_time'] = [eval_start, eval_start+500]

test_env = create_env(eval_params)
obs_space = test_env.observation_space  # get the observation space, we need this to construct our agent(s) observation input
act_space = test_env.action_space  # get the action space, we need this to construct our agent(s) action output


"""
Define training process. Ray/Tune will call this function with config params.
"""
def coop_train_fn(config, reporter):

    # initialize PPO agent on environment. This may create 1 or more workers.
    agent1 = PPOTrainer(env=env_name, config=config)

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
            reward = 0
            while not done:
                # for each observation, let the policy decides what to do
                act = agent1.compute_action(obs)
                # forward 1 step with agent action
                obs, r, done, _ = test_env.step(act)
                reward += r
            end_time = time.time()
            ep_time = end_time-start_time
            print("\n Episode time is ")
            print(ep_time)
            print("\n")
            # plot the result. This will be saved in ./results
            test_env.plot(pycigar_params['exp_tag'], env_name, i+1, reward)
    # save the params of agent
    # state = agent1.save()
    # stop the agent
    agent1.stop()


if __name__ == "__main__":
    ray.init()
    #config for RLlib - PPO agent
    config = {
        'env': env_name,
        'env_config': deepcopy(sim_params),
        "gamma": 0.5,
        'lr': 2e-04,
        'sample_batch_size': 50,
        'train_batch_size': 500,
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

"""if __name__ == "__main__":
    ray.init()
    #config for RLlib - PPO agent
    config = {
        'env': env_name,
        'env_config': deepcopy(sim_params),
        'vtrace': True,
        "gamma": 0.5,
        'lr': 5e-04,
        'lambda': 0.5,
        'sample_batch_size': 64,
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
    tune.run(coop_train_fn, config=config)"""
"""if __name__ == "__main__":
    ray.init()
    #config for RLlib - PPO agent
    config = {
        'env': env_name,
        'env_config': deepcopy(sim_params),
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        #"v_min": -10.0,  # no meaning if num_atoms == 1
        #"v_max": 10.0,   # no meaning if num_atoms == 1
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": True,
        # Whether to use double dqn
        "double_q": True,
        # Postprocess model outputs with these hidden layers to compute the
        # state and action values. See also the model config in catalog.py.
        "hiddens": [128, 64, 32],
        # N-step Q learning
        "n_step": 1,

        # === Exploration ===
        # Max num timesteps for annealing schedules. Exploration is annealed from
        # 1.0 to exploration_fraction over this number of timesteps scaled by
        # exploration_fraction
        "schedule_max_timesteps": 1000,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": 100,
        # Fraction of entire training period over which the exploration rate is
        # annealed
        "exploration_fraction": 0.1,
        # Final value of random action probability
        "exploration_final_eps": 0.02,
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 25,
        # Use softmax for sampling actions. Required for off policy estimation.
        "soft_q": False,
        # Softmax temperature. Q values are divided by this value prior to softmax.
        # Softmax approaches argmax as the temperature drops to zero.
        "softmax_temp": 1.0,
        # If True parameter space noise will be used for exploration
        # See https://blog.openai.com/better-exploration-with-parameter-noise/
        "parameter_noise": False,
        # Extra configuration that disables exploration.
        "evaluation_config": {
            "exploration_fraction": 0,
            "exploration_final_eps": 0,
        },

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 5000,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": True,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Fraction of entire training period over which the beta parameter is
        # annealed
        "beta_annealing_fraction": 0.2,
        # Final value of beta
        "final_prioritized_replay_beta": 0.4,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,
        # Whether to LZ4 compress observations
        "compress_observations": False,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 5e-4,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_norm_clipping": 40,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 50,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "sample_batch_size": 32,
        # Size of a batched sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 32,

        # === Parallelism ===
        # Number of workers for collecting samples with. This only makes sense
        # to increase if your environment is particularly slow to sample, or if
        # you"re using the Async or Ape-X optimizers.
        "num_workers": 0,
        # Whether to use a distribution of epsilons across workers for exploration.
        "per_worker_exploration": False,
        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 1,
        #'model': {'conv_filters': None, 'conv_activation': 'tanh',
        #          'fcnet_activation': 'tanh', 'fcnet_hiddens': [256, 128, 64, 32],
        #          'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True,
        #          'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256,
        #          'lstm_use_prev_action_reward': False, 'state_shape': None,
        #          'framestack': False, 'dim': 84, 'grayscale': False,
        #          'zero_mean': True, 'custom_preprocessor': None,
        #          'custom_model': None, 'custom_action_dist': None, 'custom_options': {}},
    }

    # call tune.run() to run the coop_train_fn() with the config() above
    tune.run(coop_train_fn, config=config)
"""
