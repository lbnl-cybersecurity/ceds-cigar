# import matplotlib.pyplot as plt
# plt.switch_backend('tkagg')
# from pycigar.utils.input_parser import input_parser
# import numpy as np
# from pycigar.utils.registry import register_devcon
# import tensorflow as tf
# from ray.rllib.models.catalog import ModelCatalog
# from gym.spaces import Tuple, Discrete, Box
# import matplotlib

# misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
# dss = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
# load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
# breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'

# start = 100
# hack=0.4
# sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=hack, use_load_generator=True)
# sim_params['scenario_config']['start_end_time'] = [start, start + 750]
# del sim_params['attack_randomization']
# for node in sim_params['scenario_config']['nodes']:
#     node['devices'][0]['adversary_controller'] =  'adaptive_unbalanced_fixed_controller'

# from pycigar.envs import CentralControlPhaseSpecificPVInverterEnv
# env = CentralControlPhaseSpecificPVInverterEnv(sim_params=sim_params)
# env.reset()
# done = False
# while not done:
#     _, r, done, _ = env.step([10, 10, 10])

# print('ahihi')
# from pycigar.utils.input_parser import input_parser
# import numpy as np
# from pycigar.utils.registry import register_devcon
# import tensorflow as tf
# from ray.rllib.models.catalog import ModelCatalog
# from gym.spaces import Tuple, Discrete, Box

# misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
# dss = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
# load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
# breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'


# misc_inputs_bat = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_battery/misc_inputs.csv'
# load_solar_bat = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_battery/load_solar_data.csv'
# bat = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_battery/battery_inputs.txt'

# misc_inputs = misc_inputs_bat
# dss = dss
# load_solar = load_solar_bat
# breakpoints = breakpoints
# bat = bat

# start = 100
# hack = 0.4
# sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=hack, battery_path=bat)
# sim_params['scenario_config']['start_end_time'] = [start, start + 750]
# del sim_params['attack_randomization']
# for node in sim_params['scenario_config']['nodes']:
#     node['devices'][0]['adversary_controller'] =  'adaptive_unbalanced_fixed_controller'
# start = 3601
# duration = 14400 - start

# sim_params['vectorized_mode'] = True
# sim_params['scenario_config']['start_end_time'] = [start, start + duration]
# sim_params['scenario_config']['multi_config'] = False
# sim_params['scenario_config']['custom_configs']['slack_bus_voltage'] = 1.04
# sim_params['simulation_config']['custom_configs']['solution_control_mode'] = -1

# from pycigar.envs import CentralControlPhaseSpecificPVInverterEnv
# env = CentralControlPhaseSpecificPVInverterEnv(sim_params=sim_params)
# env.reset()
# done = False
# while not done:
#     _, r, done, _ = env.step([10, 10, 10])


# from pycigar.utils.input_parser import input_parser
# import numpy as np
# import matplotlib.pyplot as plt
# plt.switch_backend('tkagg')
# from pycigar.utils.registry import register_devcon
# import tensorflow as tf
# from ray.rllib.models.catalog import ModelCatalog
# from gym.spaces import Tuple, Discrete, Box
# import random 

# misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/misc_inputs.csv'
# dss = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/ieee123.dss'
# load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/load_solar_data.csv'
# breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/breakpoints.csv'

# start = 11000

# sim_params = input_parser(misc_inputs, dss, load_solar,
#                           breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=0.3)
# sim_params['scenario_config']['start_end_time'] = [start, start + 750]
# sim_params['env_config']['sim_per_step'] = 30
# del sim_params['attack_randomization']
# for node in sim_params['scenario_config']['nodes']:
#     node['devices'][0]['adversary_controller'] =  'oscillation_fixed_controller'

# sim_params['M'] = 150
# sim_params['N'] = 2
# sim_params['P'] = 1
# sim_params['Q'] = 1
# sim_params['T'] = 150

# sim_params['cluster'] = {'1': ['s52a', 's53a', 's55a', 's56b', 's58b', 's59b', 's1a', 's2b', 's4c', 's5c', 's6c', 's7a', 's9a', 's10a', 's11a', 's12b', 's16c', 's17c', 's34c', 's19a', 's20a', 's22b', 's24c', 's28a', 's29a', 's30c', 's31c', 's32c', 's33a', 's35a', 's37a', 's38b', 's39b', 's41c', 's42a', 's43b', 's45a', 's46a', 's47', 's48', 's49a', 's49b', 's49c', 's50c', 's51a'],
#                              '2': ['s86b', 's87b', 's88a', 's90b', 's92c', 's94a', 's95b', 's96b', 's102c', 's103c', 's104c', 's106b', 's107b', 's109a', 's111a', 's112a', 's113a', 's114a', 's60a', 's62c', 's63a', 's64b', 's65a', 's65b', 's65c', 's66c', 's68a', 's69a', 's70a', 's71a', 's73c', 's74c', 's75c', 's76a', 's76b', 's76c', 's77b', 's79a', 's80b', 's82a', 's83c', 's84c', 's85c', 's98a', 's99b', 's100c']}



# from pycigar.envs.multiagent.multi_envs import ClusterMultiEnv
# env = ClusterMultiEnv(sim_params=sim_params)
# env.reset()
# done = False
# t = 0
# while not done:
#     t += 1
#     if t > 10:
#         _, r, done, _ = env.step({'1': [10, 10, 10], '2': [10, 10, 10]})
#     else:
#         _, r, done, _ = env.step({'1': [11, 12, 15], '2': [11, 15, 12]})
#     done = done['__all__']


# TEST FULL
# from pycigar.utils.input_parser import input_parser
# import numpy as np
# from pycigar.utils.registry import register_devcon
# import tensorflow as tf
# from ray.rllib.models.catalog import ModelCatalog
# from gym.spaces import Tuple, Discrete, Box
# from copy import deepcopy
# from pycigar.envs import SimpleEnv

# misc_inputs_path = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
# dss_path = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
# load_solar_path = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
# breakpoints_path = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'

# sim_params = {'misc_inputs_path': misc_inputs_path,
#               'dss_path': dss_path,
#               'load_solar_path': load_solar_path,
#               'breakpoints_path': breakpoints_path,
#               'sim_per_step': 30,
#               'mode': 'fix',
#               'hack_percentage': 0.3,
#               'start_time': 100,
#               'log': True,
#               'M': 100,
#               'N': 10,
#               'P': 1,
#               'Q': 10}

# env = SimpleEnv(sim_params)
# obs = env.reset()
# act = {}
# done = False
# t = 0
# while not done:
#     obs, reward, done, info = env.step([10, 10, 10])

# import matplotlib.pyplot as plt
# plt.switch_backend('tkagg')
# from pycigar.utils.input_parser import input_parser
# import numpy as np
# from pycigar.utils.registry import register_devcon
# import tensorflow as tf
# from ray.rllib.models.catalog import ModelCatalog
# from gym.spaces import Tuple, Discrete, Box
# import matplotlib

# misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
# dss = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'

# load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
# breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'

# start = 100
# hack = 0
# sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=hack)
# sim_params['scenario_config']['start_end_time'] = [start, start + 750]
# sim_params['env_config']['sims_per_step'] = 1
# sim_params['scenario_config']['custom_configs']['solar_scaling_factor'] = 0
# del sim_params['attack_randomization']
# for node in sim_params['scenario_config']['nodes']:
#     node['devices'][0]['adversary_controller'] =  'adaptive_unbalanced_fixed_controller'

# from pycigar.envs import CentralControlPhaseSpecificPVInverterEnv
# env = CentralControlPhaseSpecificPVInverterEnv(sim_params=sim_params)

# qv_matrix = {}
# #base voltage 
# env.reset()
# env.step([10, 10, 10])
# base_voltage = env.unwrapped.k.kernel_api.puvoltage
# for device_id in env.unwrapped.k.device.devices.keys():
#     if 'adversary' not in device_id and 'reg' not in device_id:
#         env.reset()
#         node_id = env.unwrapped.k.device.devices[device_id]['node_id']
#         env.unwrapped.k.node.nodes[node_id]['PQ_injection']['Q'] = 1
#         env.step([10, 10, 10])
#         new_voltage = env.unwrapped.k.kernel_api.puvoltage
#         qv_matrix[device_id] =  np.array(new_voltage) - np.array(base_voltage)


import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
from pycigar.utils.input_parser import input_parser
import numpy as np
from pycigar.utils.registry import register_devcon
import tensorflow as tf
from ray.rllib.models.catalog import ModelCatalog
from gym.spaces import Tuple, Discrete, Box
import matplotlib
from pycigar.utils.output import plot_new
from pycigar.utils.registry import make_create_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

misc_inputs = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
dss = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
load_solar = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
breakpoints = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'

# misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/misc_inputs.csv'
# dss = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee123busdata/ieee123.dss'
# load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/load_solar_data.csv'
# breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee123busdata/breakpoints.csv'

#best_dir = '/home/toanngo/result_lstm_ieee37_auto/sytoanngo/ieee37_lstm/main/run_train_2021-05-18_04-15-12/run_train_50674_00002_2_T=0,lr=0.0005_2021-05-18_04-15-12/latest/policy_499'
#tf.compat.v1.enable_eager_execution()
#policy = tf.saved_model.load(str(best_dir))
#infer = policy.signatures['serving_default']
#action_dist, _ = ModelCatalog.get_action_dist(Tuple([Discrete(21)] * 3), config={}, dist_type=None, framework='tf')


checkpoint = '/home/toanngo/result_lstm_ieee37_auto/sytoanngo/ieee37_lstm/main/run_train_2021-05-18_04-15-12/run_train_50674_00002_2_T=0,lr=0.0005_2021-05-18_04-15-12/checkpoint/checkpoint_499/checkpoint-499'

start = 100
hack = 0.3
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=hack)
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
sim_params['attack_randomization']['generator'] = 'AttackGeneratorEvaluation''
for node in sim_params['scenario_config']['nodes']:
    node['devices'][0]['adversary_controller'] =  'unbalanced_fixed_controller' #'unbalanced_auto_fixed_controller' #'adaptive_auto_fixed_controller'

sim_params['M'] = 300  # oscillation penalty
sim_params['N'] = 0.5  # initial action penalty
sim_params['P'] = 1    # last action penalty
sim_params['Q'] = 1    # power curtailment penalty
sim_params['T'] = 300  # imbalance penalty

pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                  'env_name': 'MultiAttackCentralControlPhaseSpecificPVInverterEnv',
                  'simulator': 'opendss'}

create_env, env_name = make_create_env(pycigar_params, version=0)
register_env(env_name, create_env)


# base_config = {
#     "env": env_name,
#     "gamma": 0.5,
#     'lr': 1e-3,
#     #"lr_schedule": [[0, 2e-2], [20000, 1e-4]],
#     'env_config': sim_params,
#     'rollout_fragment_length': 20,
#     'train_batch_size': 500, #256, #250
#     'clip_param': 0.1,
#     'lambda': 0.95,
#     'vf_clip_param': 100,

#     'num_workers': 1,
#     'num_cpus_per_worker': 1,
#     'num_cpus_for_driver': 1,
#     'num_envs_per_worker': 1,

#     'log_level': 'WARNING',

#     'model': {
#         'fcnet_activation': 'tanh',
#         'fcnet_hiddens': [32, 32], #[16, 16],
#         'free_log_std': False,
#         'vf_share_layers': False,
#         'use_lstm': True,
#         'lstm_cell_size': 16,
#         'max_seq_len': 5,
#     },

#     # ==== EXPLORATION ====
#     'explore': True,
#     'exploration_config': {
#         'type': 'StochasticSampling',  # default for PPO
#     }}

# import ray
# ray.init(local_mode=True)
# agent = PPOTrainer(config=base_config)
# agent.restore(checkpoint)

from pycigar.envs import MultiAttackCentralControlPhaseSpecificPVInverterEnv
env = MultiAttackCentralControlPhaseSpecificPVInverterEnv(sim_params=sim_params)

ob = env.reset()
ob = ob.tolist()
# act = [10, 10, 10]
# state = [[0]*16, [0]*16]
done = False

# while not done:
#     act, state, _ = agent.compute_action(ob, state=[[0]*16, [0]*16], prev_action=act, prev_reward=0)
#     ob, _, done, _ = env.step(act)
#     ob = ob.tolist()

done = False
while not done:
    _, _, done, _ = env.step([10, 10, 10])


