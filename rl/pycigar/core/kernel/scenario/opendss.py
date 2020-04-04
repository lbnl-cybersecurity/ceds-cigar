from pycigar.core.kernel.scenario import KernelScenario
from pycigar.devices import PVDevice
from pycigar.devices import RegulatorDevice

from pycigar.controllers import AdaptiveInverterController
from pycigar.controllers import FixedController
from pycigar.controllers import MimicController
from pycigar.controllers import AdaptiveFixedController
from pycigar.controllers import UnbalancedFixedController

from pycigar.controllers import RLController
import os
import numpy as np
import random
from scipy.interpolate import interp1d
import pycigar.config as config
import pandas as pd
from pycigar.envs.attack_definition import AttackDefinitionGenerator

class OpenDSSScenario(KernelScenario):

    def __init__(self, master_kernel):
        """See parent class."""
        KernelScenario.__init__(self, master_kernel)
        self.hack_start_times = None
        self.hack_end_times = None

        # take the first snapshot of randomization function if multi_config False
        self.snapshot_randomization = {}
        
    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def start_scenario(self):
        """Initialize the scenario."""
        if self.master_kernel.sim_params['scenario_config']['start_end_time'] is list:
            direct_yaml = True
        else: 
            direct_yaml = False
        start_time = self.master_kernel.sim_params['scenario_config']['start_time']
        end_time = self.master_kernel.sim_params['scenario_config']['end_time'] 

        sim_params = self.master_kernel.sim_params

        # loading attack def generator
        if 'attack_randomization' in sim_params:
            if sim_params['attack_randomization']['generator'] == 'AttackDefinitionGenerator':
                attack_def_gen = AttackDefinitionGenerator(start_time, end_time)
        else:
            attack_def_gen = None

        # load simulation and opendss file
        #network_model_directory_path = os.path.join(config.DATA_DIR, sim_params['simulation_config']['network_model_directory'])
        network_model_directory_path = sim_params['simulation_config']['network_model_directory']
        self.kernel_api.simulation_command('Redirect ' + network_model_directory_path)

        if 'solution_mode' in sim_params['simulation_config']:
            self.kernel_api.set_solution_mode(sim_params['simulation_config']['solution_mode'])
        if 'solution_number' in sim_params['simulation_config']:
            self.kernel_api.set_solution_number(sim_params['simulation_config']['solution_number'])
        if 'solution_step_size' in sim_params['simulation_config']:
            self.kernel_api.set_solution_step_size(sim_params['simulation_config']['solution_step_size'])
        if 'solution_control_mode' in sim_params['simulation_config']:
            self.kernel_api.set_solution_control_mode(sim_params['simulation_config']['solution_control_mode'])
        if 'solution_max_control_iterations' in sim_params['simulation_config']:
            self.kernel_api.\
             set_solution_max_control_iterations(sim_params['simulation_config']['solution_max_control_iterations'])

        if 'solution_max_iterations' in sim_params['simulation_config']:
            self.kernel_api.set_solution_max_iterations(sim_params['simulation_config']['solution_max_iterations'])

        self.kernel_api.set_slack_bus_voltage(sim_params['scenario_config']['custom_configs']['slack_bus_voltage'])

        # start node
        self.master_kernel.node.start_nodes()

        # create dict for hack
        self.hack_start_times = {}
        self.hack_end_times = {}

        # load device, load node and internal value for device
        for node in sim_params['scenario_config']['nodes']:
            if 'devices' in node:
                for device in node['devices']:
                    if device['type'] == 'pv_device':
                        device_type = PVDevice
                    if 'controller' in device:
                        if device['controller'] == 'adaptive_inverter_controller':
                            device_controller = AdaptiveInverterController
                        elif device['controller'] == 'rl_controller':
                            device_controller = RLController
                        elif device['controller'] == 'fixed_controller':
                            device_controller = FixedController
                        elif device['controller'] == 'adaptive_fixed_controller':
                            device_controller = AdaptiveFixedController
                        elif device['controller'] == 'unbalanced_fixed_controller':
                            device_controller = UnbalancedFixedController

                        device_configs = device['custom_configs']
                    else:
                        device_controller = AdaptiveInverterController
                        device_configs = {}

                    if 'adversary_controller' in device:
                        if device['adversary_controller'] == 'adaptive_inverter_controller':
                            adversary_device_controller = AdaptiveInverterController
                        elif device['adversary_controller'] == 'rl_controller':
                            adversary_device_controller = RLController
                        elif device['adversary_controller'] == 'fixed_controller':
                            adversary_device_controller = FixedController
                        elif device['adversary_controller'] == 'adaptive_fixed_controller':
                            adversary_device_controller = AdaptiveFixedController
                        elif device['adversary_controller'] == 'unbalanced_fixed_controller':
                            adversary_device_controller = UnbalancedFixedController
                            
                        adversary_device_configs = device['adversary_custom_configs']
                        if sim_params['tune_search'] is True:
                            adversary_device_configs = sim_params['hack_setting']
                    else:
                        adversary_device_controller = FixedController
                        adversary_device_configs = {}

                    if attack_def_gen:
                        dev_hack_info = attack_def_gen.new_dev_hack_info()
                    else:
                        dev_hack_info = device['hack']

                    if sim_params['scenario_config']['multi_config']  is False:
                        if device['name'] not in self.snapshot_randomization.keys():
                            self.snapshot_randomization[device['name']] = dev_hack_info 
                        else:
                            dev_hack_info = self.snapshot_randomization[device['name']]

                    adversary_id = self.master_kernel.device.add(name=device['name'],
                                                                 connect_to=node['name'],
                                                                 device=(device_type, device_configs),
                                                                 controller=(device_controller, device_configs),
                                                                 adversary_controller=(adversary_device_controller,
                                                                                       adversary_device_configs),
                                                                 hack=dev_hack_info)

                    # at hack start timestep, add the adversary_controller id
                    if dev_hack_info[0] in self.hack_start_times:
                        self.hack_start_times[dev_hack_info[0]].append(adversary_id)
                    else:
                        self.hack_start_times[dev_hack_info[0]] = [adversary_id]

                    # at hack end timestep, remove the adversary_controller id. See self.update()
                    # if dev_hack_info contains end_time, it's at index 2
                    if len(dev_hack_info) == 3:
                        if dev_hack_info[2] in self.hack_end_times:
                            self.hack_end_times[dev_hack_info[2]].append(adversary_id)
                        else:
                            self.hack_end_times[dev_hack_info[2]] = [adversary_id]

        # adding regulator, hotfix
        regulator_names = self.kernel_api.get_all_regulator_names()
        device_configs = sim_params['scenario_config']['regulators']
        device_configs['kernel_api'] = self.kernel_api
        for regulator_id in regulator_names:
            self.master_kernel.device.add(name=regulator_id,
                                         connect_to=None,
                                         device=(RegulatorDevice, device_configs),
                                         controller=None, 
                                         adversary_controller=None, 
                                         hack=None)

        if direct_yaml is True:
            self.change_load_profile(start_time, end_time)
        else:
            self.change_load_profile_new(start_time, end_time)
    
    def update(self, reset):
        """See parent class."""
        for node in self.master_kernel.node.nodes:
            self.master_kernel.node.nodes[node]['voltage'][self.master_kernel.time] = self.kernel_api.\
                                                                                      get_node_voltage(node)
            self.master_kernel.node.nodes[node]['PQ_injection']['P'] = 0
            self.master_kernel.node.nodes[node]['PQ_injection']['Q'] = 0
            self.master_kernel.node.total_power_inject = 0

        # hack happens here
        if self.master_kernel.time in self.hack_start_times:
            adversary_ids = self.hack_start_times[self.master_kernel.time]
            for adversary_id in adversary_ids:
                device = self.master_kernel.device.devices[adversary_id]

                temp = device['controller']
                device['controller'] = device['hack_controller']
                device['hack_controller'] = temp

        # hack stops here
        if self.master_kernel.time in self.hack_end_times:
            adversary_ids = self.hack_end_times[self.master_kernel.time]
            for adversary_id in adversary_ids:
                device = self.master_kernel.device.devices[adversary_id]
                # swapping it back
                temp = device['controller']
                device['controller'] = device['hack_controller']
                device['hack_controller'] = temp

    def change_load_profile(self, start_time, end_time):
        """Change load, solar generation at each node.

        This method is used at scenario reset.

        In the simulation configuration file, if a node has a specific
        load profile name (e.g. node_22_pv_10_minute), it will be used for
        all episodes, otherwise the load profile of the node will be set to
        a random load profile in the load profile pool.

        The solar generation profile of the node will be set to a random
        solar generation profile in the solar generation profile pool.
        """
        # scenario load profile
        sim_params = self.master_kernel.sim_params
        load_scaling_factor = sim_params['scenario_config']['custom_configs']['load_scaling_factor']

        network_data_directory_path = os.path.join(config.DATA_DIR, sim_params['scenario_config']['network_data_directory'])
        scenario_profile = [file[:-4] for file in os.listdir(network_data_directory_path) if file.endswith('.csv')]

        profile = {}
        for file in scenario_profile:
            profile[file] = np.genfromtxt(os.path.join(
                            network_data_directory_path, file + '.csv'), delimiter=',')

        list_node = self.master_kernel.node.get_node_ids()
        map_node_profile = {}
        for node in sim_params['scenario_config']['nodes']:
            node_id = node['name']
            if 'load_profile' in node and node['load_profile'] is not None:
                map_node_profile[node_id] = node['load_profile']

                load = self.interpolate_data(
                    profile[node['load_profile']][:, 3] * load_scaling_factor, start_time, end_time)
                self.master_kernel.node.set_node_load(node_id, load)
                list_node.remove(node_id)

        for node_id in list_node:
            load = self.interpolate_data(
                    random.choice(list(profile.items()))[1][:, 3]*load_scaling_factor, start_time, end_time)
            self.master_kernel.node.set_node_load(node_id, load)

        # scenario solar generation profile for PV Device
        solar_scaling_factor = sim_params['scenario_config']['custom_configs']['solar_scaling_factor']

        list_pv_device_ids = self.master_kernel.device.get_pv_device_ids()

        for device_id in list_pv_device_ids:
            if 'adversary' not in device_id:
                node_id = self.master_kernel.device.get_node_connected_to(device_id)
                percentage_control = self.master_kernel.device.get_device(device_id).percentage_control
                solar = self.interpolate_data(profile[map_node_profile[node_id]][:, 1] *
                                              solar_scaling_factor, start_time, end_time)*percentage_control
                self.master_kernel.device.set_device_internal_scenario(device_id, solar)

                device_id = 'adversary_' + device_id
                node_id = self.master_kernel.device.get_node_connected_to(device_id)
                percentage_control = self.master_kernel.device.get_device(device_id).percentage_control
                solar = self.interpolate_data(profile[map_node_profile[node_id]][:, 1] *
                                              solar_scaling_factor, start_time, end_time) * percentage_control
                self.master_kernel.device.set_device_internal_scenario(device_id, solar)

    def interpolate_data(self, data, start_time, end_time):
        """Interpolate data.

        This is used to interpolate load and solar generation to have more
        data points between start time and end time
        (from minute resolution to second resolution).

        Parameters
        ----------
        data : list
            The raw data, load data with minute resolution.
        start_time : int
            The second of the day which we want to start
        end_time : int
            The second of the day which we want to stop, end_time > start_time

        Returns
        -------
        list
            The data after interpolating.
        """
        # Interpolate to get minutes to seconds
        t_seconds = np.linspace(1, len(data), int(3600*24/1))
        f = interp1d(range(len(data)), data, kind='cubic', fill_value="extrapolate")
        data_secs = f(t_seconds)
        return data_secs[start_time:end_time]

    def change_load_profile_new(self, start_time, end_time):
        sim_params = self.master_kernel.sim_params
        load_scaling_factor = sim_params['scenario_config']['custom_configs']['load_scaling_factor']

        network_data_directory_path = os.path.join(config.DATA_DIR, sim_params['scenario_config']['network_data_directory'])

        profile = pd.read_csv(network_data_directory_path)
        profile.columns = map(str.lower, profile.columns)
        list_node = self.master_kernel.node.get_node_ids()

        for node in sim_params['scenario_config']['nodes']:
            node_id = node['name']
            load = np.array(profile[node_id])[start_time:end_time] * load_scaling_factor
            self.master_kernel.node.set_node_load(node_id, load)
        solar_scaling_factor = sim_params['scenario_config']['custom_configs']['solar_scaling_factor']
        list_pv_device_ids = self.master_kernel.device.get_pv_device_ids()

        for device_id in list_pv_device_ids:
            if 'adversary' not in device_id:
                node_id = self.master_kernel.device.get_node_connected_to(device_id)
                percentage_control = self.master_kernel.device.get_device(device_id).percentage_control
                solar = np.array(profile[node_id + '_pv'])[start_time:end_time]*solar_scaling_factor*percentage_control
                self.master_kernel.device.set_device_internal_scenario(device_id, solar)

                device_id = 'adversary_' + device_id
                node_id = self.master_kernel.device.get_node_connected_to(device_id)
                percentage_control = self.master_kernel.device.get_device(device_id).percentage_control
                solar = np.array(profile[node_id + '_pv'])[start_time:end_time]*solar_scaling_factor*percentage_control
                self.master_kernel.device.set_device_internal_scenario(device_id, solar)
