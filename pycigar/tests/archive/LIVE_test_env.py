from pycigar.core.kernel.kernel import Kernel
from pycigar.core.kernel.device import KernelDevice
from pycigar.devices import PVDevice
from pycigar.controllers import AdaptiveInverterController
from pycigar.controllers import FixedController
from pycigar.controllers import RLController
import matplotlib.pyplot as plt
from pycigar.envs.pv_inverter_env import PVInverterEnv
# load params from .ini or .yaml
import yaml
import os
import numpy as np
stream = open("pseudo_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)
import timeit

#Your statements here

env = PVInverterEnv(sim_params)
state = env.reset()
done = False

start = timeit.default_timer()
count  = 0
voltage = []
solar = []
control = {}

ls = ['dl_82034', 'dl_858m', 'dload_806m', 'dload_810', 'dload_810m', 'dload_822',
'dload_822m', 'dload_824', 'dload_826', 'dload_826m', 'dload_828', 'dload_830',
'dload_830m', 'dload_834', 'dload_836', 'dload_838', 'dload_838m', 'dload_840',
'dload_844', 'dload_846', 'dload_848', 'dload_856', 'dload_856m', 'dload_860',
'dload_864', 'dload_864m', 'sload_840', 'sload_844', 'sload_848', 'sload_860', 'sload_890']

sample_node = 'dload_824'
while not done:
    next_observation, reward, done, infos = env.step(np.array([0.9, 1.0, 1.0, 1.1, 1.2]))
    #
    #print(env.k.node.nodes['dl_82034']['voltage'], file=sys.stderr)
    #print(env.k.device.devices['adversary_pv_' + str(8)])
    vol = env.k.kernel_api.get_node_voltage(sample_node)
    sol = env.k.device.devices['pv_1']['device'].solar_generation[count]
    for i in range(31):
        col = env.k.device.devices['pv_' + str(i+1)]['device'].control_setting
        if i not in control.keys():
            control[i] = [col]
        else:
            control[i].append(col)
    #col = env.k.device.devices['pv_1']['device'].control_setting
    #control.append(col)

    voltage.append(vol)
    solar.append(sol)
    count += 1

"""f = plt.figure(figsize=[20, 10])
plt.plot(solar)
plt.title('Voltage at Inverter Nodes')
plt.show()"""

"""f = plt.figure(figsize=[20, 10])
plt.plot(voltage)
plt.title('Voltage at Inverter Nodes')
plt.show()"""

f = plt.figure(figsize=[20, 10])
i = 5
#print(control[i])
plt.plot(control[i])
plt.show()

#stop = timeit.default_timer()
#print('Time total this step: ', stop - start, file=sys.stderr)
