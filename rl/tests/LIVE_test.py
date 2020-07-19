from pycigar.core.kernel.kernel import Kernel
from pycigar.core.kernel.device import KernelDevice
from pycigar.devices import PVDevice
from pycigar.controllers import AdaptiveInverterController
from pycigar.controllers import FixedController
from pycigar.controllers import RLController

# load params from .ini or .yaml
import yaml
import os
stream = open("pseudo_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

k = Kernel(simulator="opendss", sim_params=sim_params)

# start the simulation, get the api to interact with simulator
kernel_api = k.simulation.start_simulation()

k.pass_api(kernel_api)

k.scenario.start_scenario()

print(k.node.nodes['dl_82034']['load'])
k.update(reset=True)
k.update(reset=False)
k.update(reset=False)
#kernel_api.simulation_command("Redirect /home/toanngo/Documents/GitHub/power/tests/feeder34_test.dss")
#k.pass_api(kernel_api)
#print(k.node.start_nodes())
#print(k.node.nodes['dl_82034'])
#print(kernel_api.get_node_ids())
#k.device.add(
#    connect_to='dl_82034', controller=(AdaptiveInverterController, {})
#    )
#print(k.device.devices['adversary_device_0']['controller'].device_id)
#kernel_api.set_node_kw("dload_810m", 7.0)
#kernel_api.set_node_kvar("dload_810m", 7.0)
#rint(kernel_api.get_node_voltage("dload_810m"))
#print(kernel_api.check_simulation_converged())
#kernel_api.simulation_command("R")
#attentive interface
#k.scenario.start_scenario(sim_params=sim_params) #load
#test node start

k.simulation.teardown_opendss()
