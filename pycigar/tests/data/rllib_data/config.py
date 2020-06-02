from copy import deepcopy

import pycigar
from pycigar.utils.input_parser import input_parser
from pycigar.utils.registry import make_create_env
from ray.tune.registry import register_env

pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                  'env_name': 'CentralControlPVInverterEnv',
                  'simulator': 'opendss'}

create_env, env_name = make_create_env(pycigar_params, version=0)
register_env(env_name, create_env)

misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)

base_config = {
    "env": env_name,
    "gamma": 0.5,
    'lr': 2e-4,
    'env_config': deepcopy(sim_params),
    'rollout_fragment_length': 50,
    'train_batch_size': 500,
    'clip_param': 0.1,
    'lambda': 0.95,
    'vf_clip_param': 100,

    'num_workers': 8,
    'num_cpus_per_worker': 1,
    'num_cpus_for_driver': 1,
    'num_envs_per_worker': 1,

    'log_level': 'WARNING',

    'model': {
        'fcnet_activation': 'tanh',
        'fcnet_hiddens': [128, 64, 32],
        'free_log_std': False,
        'vf_share_layers': True,
        'use_lstm': False,
        'state_shape': None,
        'framestack': False,
        'zero_mean': True,
    },

    # ==== EXPLORATION ====
    'explore': True,
    'exploration_config': {
        'type': 'StochasticSampling',  # default for PPO
    },

    # ==== EVALUATION ====
    "evaluation_num_workers": 1,
    'evaluation_num_episodes': 1,
    "evaluation_interval": 5,
    'evaluation_config': {
        "seed": 42,
        # IMPORTANT NOTE: For policy gradients, this might not be the optimal policy
        'explore': False,
        'env_config': deepcopy(sim_params),
    },
}
# eval environment should not be random across workers
eval_start = 100  # random.randint(0, 3599 - 500)
base_config['evaluation_config']['env_config']['scenario_config']['start_end_time'] = [eval_start, eval_start + 750]
base_config['evaluation_config']['env_config']['scenario_config']['multi_config'] = False
del base_config['evaluation_config']['env_config']['attack_randomization']
