from pycigar.envs.multiagent import MultiIACEnv
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np
stream = open("sanity_check_pseudo_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

env = MultiIACEnv(sim_params)
state = env.reset()

all_loads = ['dl_82034', 'dl_858m', 'dload_806m', 'dload_810', 'dload_810m', 'dload_822', 'dload_822m', 'dload_824', 'dload_826', 'dload_826m', 'dload_828', 'dload_830', 'dload_830m', 'dload_834', 'dload_836', 'dload_838', 'dload_838m', 'dload_840', 'dload_844', 'dload_846', 'dload_848', 'dload_856', 'dload_856m', 'dload_860', 'dload_864', 'dload_864m', 'sload_840', 'sload_844', 'sload_848', 'sload_860', 'sload_890']
devices = ['pv_5','pv_6','pv_7','pv_8','pv_9','pv_10','pv_11','pv_12','pv_13','pv_14','pv_15','pv_16','pv_17']

count = 0
done = False
#rl_actions = {'pv_8': np.array([0.9, 1.0, 1.0, 1.1, 1.2]),
#              'pv_9': np.array([0.9, 1.0, 1.0, 1.1, 1.2])}
rl_actions = {}
vol_value = []
y_value = []
p_value = []
R_value = []
while not done:

    next_observation, reward, done, infos = env.step(rl_actions)
    vol_value.append(reward['pv_8'][0])
    y_value.append(reward['pv_8'][1])
    p_value.append(reward['pv_8'][2])
    R_value.append(reward['pv_8'][3])
    done = done['__all__']
    count += 1

plt.plot(vol_value)
plt.show()

plt.plot(y_value)
plt.show()

plt.plot(p_value)
plt.show()
