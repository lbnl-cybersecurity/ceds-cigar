from pycigar.core.kernel.scenario import KernelScenario
from pycigar.devices import PVDevice
from pycigar.devices import RegulatorDevice
import opendssdirect as dss
from pycigar.utils.data_generation.load import LoadGenerator
from pycigar.controllers import AdaptiveInverterController
from pycigar.controllers import FixedController
from pycigar.controllers import AdaptiveFixedController
from pycigar.controllers import UnbalancedFixedController

from pycigar.controllers import RLController

import os
import numpy as np
import pandas as pd

from pycigar.utils.pycigar_registration import pycigar_make
from pycigar.utils.logging import logger
import random


class OpenDSSScenario(KernelScenario):
    def __init__(self, master_kernel):
        """See parent class."""
        KernelScenario.__init__(self, master_kernel)
        self.hack_start_times = None
        self.hack_end_times = None

        # take the first snapshot of randomization function if multi_config False
        self.snapshot_randomization = {}
        # capture the attack generator to make sure it is only init once
        self.attack_def_gen = None
        self.load_generator = None

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def start_scenario(self):
        """Initialize the scenario."""
        start_time = self.master_kernel.sim_params['scenario_config']['start_time']
        end_time = self.master_kernel.sim_params['scenario_config']['end_time']

        sim_params = self.master_kernel.sim_params

        if 'attack_randomization' in sim_params and not self.attack_def_gen:
            self.attack_def_gen = pycigar_make(sim_params['attack_randomization']['generator'], start_time=start_time, end_time=end_time)

        # overwrite multi_config to have a new start_time and end_time
        if self.attack_def_gen:
            start_time, end_time = self.attack_def_gen.change_mode()
            self.master_kernel.sim_params['scenario_config']['start_time'] = start_time
            self.master_kernel.sim_params['scenario_config']['end_time'] = end_time

        # load simulation and opendss file
        # network_model_directory_path = os.path.join(config.DATA_DIR, sim_params['simulation_config']['network_model_directory'])
        network_model_directory_path = sim_params['simulation_config']['network_model_directory']
        if isinstance(network_model_directory_path, str):
            network_path = network_model_directory_path
        else:
            network_path = random.sample(network_model_directory_path, 1)[0]
        self.kernel_api.simulation_command('Redirect "{}"'.format(network_path))
        additional_api_info = {}
        additional_api_info['split_phase'] = sim_params.get('split_phase', False)
        self.kernel_api.start_api(**additional_api_info)

        solution_mode = sim_params['simulation_config']['custom_configs'].get('solution_mode', 1)
        solution_number = sim_params['simulation_config']['custom_configs'].get('solution_number', 1)
        solution_step_size = sim_params['simulation_config']['custom_configs'].get('solution_step_size', 1)
        solution_control_mode = sim_params['simulation_config']['custom_configs'].get('solution_control_mode', -1)
        solution_max_control_iterations = sim_params['simulation_config']['custom_configs'].get('solution_max_control_iterations', 1000000)
        solution_max_iterations = sim_params['simulation_config']['custom_configs'].get('solution_max_iterations', 30000)

        self.kernel_api.set_solution_mode(solution_mode)
        self.kernel_api.set_solution_number(solution_number)
        self.kernel_api.set_solution_step_size(solution_step_size)
        self.kernel_api.set_solution_control_mode(solution_control_mode)
        self.kernel_api.set_solution_max_control_iterations(solution_max_control_iterations)
        self.kernel_api.set_solution_max_iterations(solution_max_iterations)


        slack_bus_voltage = sim_params['scenario_config']['custom_configs'].get('slack_bus_voltage', 1.04)
        self.kernel_api.set_slack_bus_voltage(slack_bus_voltage)

        # start node
        self.master_kernel.node.start_nodes()

        # create dict for hack
        self.hack_start_times = {}
        self.hack_end_times = {}


        if 'controllers' in sim_params['scenario_config']:
            for controller in sim_params['scenario_config']['controllers']:
                controller_name = controller.get('name', None)
                controller_type = controller.get('controller', None)
                controller_configs = controller.get('custom_controller_configs', None)
                list_devices = controller.get('list_devices', None)
                self.master_kernel.device.add_controller(
                    name=controller_name,
                    controller=(controller_type, controller_configs, list_devices)
                )

        # load device, load node and internal value for device
        for node in sim_params['scenario_config']['nodes']:
            if 'devices' in node:
                for device in node['devices']:
                    device_type = device['device']
                    device_configs = device.get('custom_device_configs', None)

                    controller = device.get('controller', None)
                    custom_controller_configs = device.get('custom_controller_configs', None)

                    adversary_controller = device.get('adversary_controller', None)
                    adversary_custom_controller_configs = device.get('adversary_custom_controller_configs', None)

                    # if there is no hack at all at the device
                    if adversary_controller is None:
                        dev_hack_info = None
                    else:
                        if self.attack_def_gen:
                            dev_hack_info = self.attack_def_gen.new_dev_hack_info()
                        else:
                            dev_hack_info = device['hack']
                        if sim_params['scenario_config']['multi_config'] is False:
                            if device['name'] not in self.snapshot_randomization.keys():
                                self.snapshot_randomization[device['name']] = dev_hack_info
                            else:
                                dev_hack_info = self.snapshot_randomization[device['name']]

                    adversary_id = self.master_kernel.device.add(
                        name=device['name'],
                        connect_to=node['name'],
                        device=(device_type, device_configs),
                        controller=(controller, custom_controller_configs),
                        adversary_controller=(adversary_controller, adversary_custom_controller_configs),
                        hack=dev_hack_info,
                    )

                    if dev_hack_info is not None and adversary_id is not None:
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
                    if dev_hack_info is not None and not self.master_kernel.sim_params['is_disable_log']:
                        Logger = logger()
                        Logger.custom_metrics[device['name']] = dev_hack_info
        # adding regulator, hotfix
        regulator_names = self.kernel_api.get_all_regulator_names()
        if regulator_names and 'regulators' in sim_params['scenario_config']:
            device_configs = sim_params['scenario_config']['regulators']
            device_configs['kernel_api'] = self.kernel_api
            for regulator_id in regulator_names:
                self.master_kernel.device.add(
                    name=regulator_id,
                    connect_to=None,
                    device=(RegulatorDevice, device_configs),
                    controller=None,
                    adversary_controller=None,
                    hack=None,
                )

        self.choose_attack = 0 #random.randrange(2)

        if dev_hack_info is not None and not self.master_kernel.sim_params['is_disable_log']:
            Logger = logger()
            Logger.custom_metrics['hack'] = dev_hack_info[1]

    def update(self, reset):
        """See parent class."""
        self.kernel_api.update_all_bus_voltages()
        for node in self.master_kernel.node.nodes:
            self.master_kernel.node.nodes[node]['voltage'][self.master_kernel.time] = self.kernel_api.get_node_voltage(
                node
            )
            if self.master_kernel.sim_params is None or not self.master_kernel.sim_params['is_disable_log']:
                Logger = logger()
                Logger.log(node, 'voltage', self.master_kernel.node.nodes[node]['voltage'][self.master_kernel.time])

            self.master_kernel.node.nodes[node]['PQ_injection']['P'] = 0
            self.master_kernel.node.nodes[node]['PQ_injection']['Q'] = 0

        # hack happens here
        if self.hack_start_times and self.master_kernel.time in self.hack_start_times:
            adversary_ids = self.hack_start_times[self.master_kernel.time]
            for adversary_id in adversary_ids:
                device = self.master_kernel.device.devices[adversary_id]

                if isinstance(device['hack_controller'], list):
                    temp = device['controller']
                    device['controller'] = device['hack_controller'][self.choose_attack]
                    device['hack_controller'] = temp
                else:
                    temp = device['controller']
                    device['controller'] = device['hack_controller']
                    device['hack_controller'] = temp

                self.master_kernel.device.update_kernel_device_info(adversary_id)

        # hack stops here
        if self.hack_end_times and self.master_kernel.time in self.hack_end_times:
            adversary_ids = self.hack_end_times[self.master_kernel.time]
            for adversary_id in adversary_ids:
                device = self.master_kernel.device.devices[adversary_id]
                # swapping it back
                temp = device['controller']
                device['controller'] = device['hack_controller']
                device['hack_controller'] = temp

                self.master_kernel.device.update_kernel_device_info(adversary_id)

    def change_load_profile(
        self, start_time, end_time, load_scaling_factor=1.5, solar_scaling_factor=1, network_data_directory_path=None,
    ):
        sim_params = self.master_kernel.sim_params
        if sim_params:
            load_scaling_factor = sim_params['scenario_config']['custom_configs']['load_scaling_factor']
            solar_scaling_factor = sim_params['scenario_config']['custom_configs']['solar_scaling_factor']
            network_data_directory_path = sim_params['scenario_config']['network_data_directory']
            use_load_generator = sim_params['scenario_config']['use_load_generator']

        profile = pd.read_csv(network_data_directory_path)
        profile.columns = map(str.lower, profile.columns)
        load_df = profile

        if sim_params and use_load_generator:
            if self.load_generator is None:
                data = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/utils/data_generation/load/data'
                data = [data + '/7_MWp_P.csv', data + '/10_MWp_P.csv', data + '/12_MWp_P.csv', data + '/19_MWp_P.csv']
                self.load_generator = LoadGenerator(data)
            df = dss.utils.loads_to_dataframe()
            load_levels = (df['kW']/1000).tolist()
            load_names = df.index.tolist()
            load_profiles = self.load_generator.generate_load(load_levels)
            profile_gen = pd.DataFrame(np.array([i.values for i in load_profiles]).T, columns=load_names)/1000 #pd.DataFrame(load_profiles, columns=load_names)/1000
            load_df = profile_gen

        for node_id in self.master_kernel.node.nodes.keys():
            if node_id in profile:
                load = np.array(load_df[node_id])[start_time:end_time] * load_scaling_factor
            else:
                load = np.zeros(end_time-start_time)
            self.master_kernel.node.set_node_load(node_id, load)

        list_pv_device_ids = self.master_kernel.device.get_pv_device_ids()

        for device_id in list_pv_device_ids:
            if 'adversary' not in device_id:
                node_id = self.master_kernel.device.get_node_connected_to(device_id)
                percentage_control = self.master_kernel.device.get_device(device_id).percentage_control
                solar = np.array(profile[node_id + '_pv'])[start_time:end_time] * solar_scaling_factor * percentage_control
                sbar = np.max(np.array(profile[node_id + '_pv']) * solar_scaling_factor * percentage_control)
                self.master_kernel.device.set_device_internal_scenario(device_id, solar)
                self.master_kernel.device.set_device_sbar(device_id, sbar)

                device_id = 'adversary_' + device_id
                if device_id in list_pv_device_ids:
                    node_id = self.master_kernel.device.get_node_connected_to(device_id)
                    percentage_control = self.master_kernel.device.get_device(device_id).percentage_control
                    solar = np.array(profile[node_id + '_pv'])[start_time:end_time] * solar_scaling_factor * percentage_control
                    sbar = np.max(np.array(profile[node_id + '_pv']) * solar_scaling_factor * percentage_control)
                    self.master_kernel.device.set_device_internal_scenario(device_id, solar)
                    self.master_kernel.device.set_device_sbar(device_id, sbar)
