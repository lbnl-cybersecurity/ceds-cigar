import os
import random

import numpy as np
import pycigar.config as config
from pycigar.controllers import AdaptiveInverterController
from pycigar.controllers import FixedController
from pycigar.controllers import FixedRegController, BaseRegController, BaseLineController, FixedLineController
from pycigar.controllers import RLController
from pycigar.core.kernel.scenario import KernelScenario
from pycigar.devices import PVDevice
from pycigar.devices import RegulatorDevice, LineDevice
from scipy.interpolate import interp1d


class OpenDSSScenario(KernelScenario):

    def __init__(self, master_kernel):
        """See parent class."""
        KernelScenario.__init__(self, master_kernel)

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def start_scenario(self):
        """Initialize the scenario."""
        start_time, end_time = self.master_kernel.sim_params['scenario_config']['start_end_time']
        sim_params = self.master_kernel.sim_params

        # load simulation and opendss file
        network_model_directory_path = os.path.join(config.DATA_DIR,
                                                    sim_params['simulation_config']['network_model_directory'])
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
            self.kernel_api.set_solution_max_control_iterations(sim_params['simulation_config']['solution_max_control_iterations'])

        if 'solution_max_iterations' in sim_params['simulation_config']:
            self.kernel_api.set_solution_max_iterations(sim_params['simulation_config']['solution_max_iterations'])

        self.kernel_api.set_slack_bus_voltage(sim_params['scenario_config']['custom_configs']['slack_bus_voltage'])

        # start node
        self.master_kernel.node.start_nodes()

        # create dict for hack
        self.hack_time = {}

        # load device, load node and internal value for device
        for node in sim_params['scenario_config']['nodes']:
            if 'devices' in node:
                for device in node['devices']:
                    if device['type'] == 'pv_device':
                        device_type = PVDevice
                        # print(device)
                    if 'controller' in device:
                        if device['controller'] == 'adaptive_inverter_controller':
                            device_controller = AdaptiveInverterController
                        elif device['controller'] == 'rl_controller':
                            device_controller = RLController
                        elif device['controller'] == 'fixed_controller':
                            device_controller = FixedController
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
                        adversary_device_configs = device['adversary_custom_configs']
                        if sim_params['tune_search'] is True:
                            adversary_device_configs = sim_params['hack_setting']
                    else:
                        adversary_device_controller = FixedController
                        adversary_device_configs = {}
                    adversary_id = self.master_kernel.device.add(name=device['name'],
                                                                 connect_to=node['name'],
                                                                 device=(device_type, device_configs),
                                                                 controller=(device_controller, device_configs),
                                                                 adversary_controller=(adversary_device_controller,
                                                                                       adversary_device_configs),
                                                                 hack=device['hack'], device_type='pv_device')

                    # at hack timestep, add the adversary_controller id
                    if device['hack'][0] in self.hack_time:
                        self.hack_time[device['hack'][0]].append(adversary_id)
                    else:
                        self.hack_time[device['hack'][0]] = [adversary_id]

        if 'regs' in sim_params['scenario_config'].keys():
            for reg in sim_params['scenario_config']['regs']:
                if 'devices' in reg:
                    for device in reg['devices']:
                        adversary_device_configs = dict()
                        if device['type'] == 'reg_device':
                            device_type = RegulatorDevice

                        if 'controller' in device:
                            if device['controller'] == 'base_reg_controller':
                                device_controller = BaseRegController

                        if 'adversary_controller' in device:
                            if device['adversary_controller'] == 'fixed_reg_controller':
                                adversary_device_controller = FixedRegController
                            adversary_device_configs = device['adversary_custom_configs']
                        adversary_id = self.master_kernel.device.add(name=device['name'],
                                                                     connect_to=reg['name'],
                                                                     device=(device_type, device['custom_configs']),
                                                                     controller=(
                                                                         device_controller, device['custom_configs']),
                                                                     adversary_controller=(adversary_device_controller,
                                                                                           adversary_device_configs),
                                                                     hack=device['hack'], device_type='reg_device')
                        if device['hack'][0] in self.hack_time:
                            self.hack_time[device['hack'][0]].append(adversary_id)
                        else:
                            self.hack_time[device['hack'][0]] = [adversary_id]

        else:
            print('No Regulator Found')

        if 'lines' in sim_params['scenario_config'].keys():
            for line in sim_params['scenario_config']['lines']:
                if 'devices' in line:
                    for device in line['devices']:
                        adversary_device_configs = dict()
                        if device['type'] == 'line_device':
                            device_type = LineDevice

                        if 'controller' in device:
                            if device['controller'] == 'base_line_controller':
                                device_controller = BaseLineController

                        if 'adversary_controller' in device:
                            if device['adversary_controller'] == 'fixed_line_controller':
                                adversary_device_controller = FixedLineController
                            adversary_device_configs = device['adversary_custom_configs']
                        adversary_id = self.master_kernel.device.add(name=device['name'],
                                                                     connect_to=line['name'],
                                                                     device=(device_type, device['custom_configs']),
                                                                     controller=(
                                                                         device_controller, device['custom_configs']),
                                                                     adversary_controller=(adversary_device_controller,
                                                                                           adversary_device_configs),
                                                                     hack=device['hack'], device_type='line_device')
                        if device['hack'][0] in self.hack_time:
                            self.hack_time[device['hack'][0]].append(adversary_id)
                        else:
                            self.hack_time[device['hack'][0]] = [adversary_id]
        else:
            print('No Line Change Issue Found')

    def update(self, reset):
        """See parent class."""
        for node in self.master_kernel.node.nodes:
            self.master_kernel.node.nodes[node]['voltage'][self.master_kernel.time] = self.kernel_api. \
                get_node_voltage(node)
            self.master_kernel.node.nodes[node]['PQ_injection']['P'] = 0
            self.master_kernel.node.nodes[node]['PQ_injection']['Q'] = 0

        # hack happens here
        if self.master_kernel.time in self.hack_time:
            adversary_ids = self.hack_time[self.master_kernel.time]
            for adversary_id in adversary_ids:
                device = self.master_kernel.device.devices[adversary_id]

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

        network_data_directory_path = os.path.join(config.DATA_DIR,
                                                   sim_params['scenario_config']['network_data_directory'])
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
                random.choice(list(profile.items()))[1][:, 3] * load_scaling_factor, start_time, end_time)
            self.master_kernel.node.set_node_load(node_id, load)

        # scenario solar generation profile for PV Device
        solar_scaling_factor = sim_params['scenario_config']['custom_configs']['solar_scaling_factor']

        list_pv_device_ids = self.master_kernel.device.get_pv_device_ids()

        for device_id in list_pv_device_ids:
            if 'adversary' not in device_id:
                node_id = self.master_kernel.device.get_node_connected_to(device_id)
                percentage_control = self.master_kernel.device.get_device(device_id).percentage_control
                solar = self.interpolate_data(profile[map_node_profile[node_id]][:, 1] *
                                              solar_scaling_factor, start_time, end_time) * percentage_control
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
        t_seconds = np.linspace(1, len(data), int(3600 * 24 / 1))
        f = interp1d(range(len(data)), data, kind='cubic', fill_value="extrapolate")
        data_secs = f(t_seconds)
        return data_secs[start_time:end_time]
