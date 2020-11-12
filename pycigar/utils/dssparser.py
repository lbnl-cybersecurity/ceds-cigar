import os
import pandas as pd
import re
import numpy as np


def input_parser(dss_path):    
    file_dss_path = dss_path
    
    json_query = {

        'hack_setting': {'default_control_setting': [1.039, 1.04, 1.04, 1.041, 1.042]},
        'env_config': {'clip_actions': True, 'sims_per_step': 20},
        'attack_randomization': {'generator': 'AttackDefinitionGenerator'},
        'simulation_config': {
            'network_model_directory': file_dss_path,
            'custom_configs': {
                'solution_mode': 1,
                'solution_number': 1,
                'solution_step_size': 1,
                'solution_control_mode': 2,
                'solution_max_control_iterations': 1000000,
                'solution_max_iterations': 30000,
                'power_factor': 0.9,
            },
        },
        'scenario_config': {
            'multi_config': True,
            'start_end_time': 750,
            'network_data_directory': file_load_solar_path,
            'custom_configs': {
                'load_scaling_factor': 1.5,
                'solar_scaling_factor': 3,
                'slack_bus_voltage': 1.02,  # default 1.04
                'load_generation_noise': False,
            },
            'nodes': [],
            'regulators': {'max_tap_change': 30, 'forward_band': 16, 'tap_number': 2, 'tap_delay': 0, 'delay': 30},
        },
    }

    # read misc_input
    # misc_inputs_data = pd.read_csv(file_misc_inputs_path, header=None)
    # misc_inputs_data = misc_inputs_data.T
    # new_header = misc_inputs_data.iloc[0]  # grab the first row for the header
    # misc_inputs_data = misc_inputs_data[1:]  # take the data less the header row
    # misc_inputs_data.columns = new_header  # set the header row as the df header
    # misc_inputs_data = misc_inputs_data.to_dict()

    # M = misc_inputs_data['Oscillation Penalty'][1]
    # N = misc_inputs_data['Action Penalty'][1]
    # P = misc_inputs_data['Deviation from Optimal Penalty'][1]
    # Q = misc_inputs_data['PsetPmax Penalty'][1]
    # power_factor = misc_inputs_data['power factor'][1]
    # load_scaling_factor = misc_inputs_data['load scaling factor'][1]
    # solar_scaling_factor = misc_inputs_data['solar scaling factor'][1]
    # p_ramp_rate = misc_inputs_data['p ramp rate'][1]
    # q_ramp_rate = misc_inputs_data['q ramp rate'][1]

    # json_query['M'] = M
    # json_query['N'] = N
    # json_query['P'] = P
    # json_query['Q'] = Q
    # json_query['vectorized_mode'] = vectorized_mode
    # json_query['scenario_config']['custom_configs']['load_scaling_factor'] = load_scaling_factor
    # json_query['scenario_config']['custom_configs']['solar_scaling_factor'] = solar_scaling_factor
    # json_query['scenario_config']['custom_configs']['power_factor'] = power_factor

    # low_pass_filter_measure_mean = misc_inputs_data['measurement filter time constant mean'][1]
    # low_pass_filter_measure_std = misc_inputs_data['measurement filter time constant std'][1]
    # low_pass_filter_output_mean = misc_inputs_data['output filter time constant mean'][1]
    # low_pass_filter_output_std = misc_inputs_data['output filter time constant std'][1]
    # default_control_setting = [
    #     misc_inputs_data['bp1 default'][1],
    #     misc_inputs_data['bp2 default'][1],
    #     misc_inputs_data['bp3 default'][1],
    #     misc_inputs_data['bp4 default'][1],
    #     misc_inputs_data['bp5 default'][1],
    # ]

    # read load_solar_data & read
    # load_solar_data = pd.read_csv(file_load_solar_path)
    # node_names = [node for node in list(load_solar_data) if '_pv' not in node]
    # breakpoints_data = pd.read_csv(file_breakpoints_path)

    # for node in node_names:
    #     node_default_control_setting = default_control_setting
    #     if node + '_pv' in list(breakpoints_data):
    #         node_default_control_setting = breakpoints_data[node + '_pv'].tolist()

    #     node_description = {}
    #     node_description['name'] = node.lower()
    #     node_description['load_profile'] = None
    #     node_description['devices'] = []
    #     device = {}
    #     device['name'] = 'inverter_' + node.lower()
    #     device['device'] = 'pv_device'
    #     device['custom_device_configs'] = {}
    #     device['custom_device_configs']['default_control_setting'] = node_default_control_setting
    #     device['custom_device_configs']['gain'] = 1e5
    #     device['custom_device_configs']['is_butterworth_filter'] = False
    #     device['custom_device_configs']['p_ramp_rate'] = p_ramp_rate
    #     device['custom_device_configs']['q_ramp_rate'] = q_ramp_rate
    #     device['custom_device_configs']['low_pass_filter_measure_mean'] = low_pass_filter_measure_mean
    #     device['custom_device_configs']['low_pass_filter_output_mean'] = low_pass_filter_output_mean
    #     # if not benchmark:
    #     #     device['custom_device_configs']['low_pass_filter_measure_std'] = low_pass_filter_measure_std
    #     #     device['custom_device_configs']['low_pass_filter_output_std'] = low_pass_filter_output_std

    #     # device['controller'] = 'rl_controller'
    #     # device['custom_controller_configs'] = {}
    #     # device['custom_controller_configs']['default_control_setting'] = node_default_control_setting
    #     # if norl_mode:
    #     #     device['controller'] = 'adaptive_inverter_controller'
    #     #     device['custom_controller_configs']['delay_timer'] = 60
    #     #     device['custom_controller_configs']['threshold'] = 0.05
    #     #     device['custom_controller_configs']['adaptive_gain'] = 20

    #     # device['adversary_controller'] = 'adaptive_fixed_controller'
    #     # if adv:
    #     #     device['adversary_controller'] = 'rl_controller'
    #     # device['adversary_custom_controller_configs'] = {}
    #     # device['adversary_custom_controller_configs']['default_control_setting'] = [1.014, 1.015, 1.015, 1.016, 1.017]
    #     # device['hack'] = [250, percentage_hack, 500]

    #     node_description['devices'].append(device)
    #     json_query['scenario_config']['nodes'].append(node_description)

    # json_query['scenario_config']['regulators']['max_tap_change'] = max_tap_change
    # json_query['scenario_config']['regulators']['forward_band'] = forward_band
    # json_query['scenario_config']['regulators']['tap_number'] = tap_number
    # json_query['scenario_config']['regulators']['tap_delay'] = tap_delay
    # json_query['scenario_config']['regulators']['delay'] = delay

    # ---DSS Parser Start-------------------------------------------------------------------------------

    f = open(dss_path, "r")
    input_file = f.read()
    print(input_file)

    input_list = [x.lower() for x in input_file.split("\n")]
    while ('' in input_list):
        input_list.remove('')
    print(input_list, '\n')

    device_list = []
    controller_list = []

    for i in input_list:

        if i[0:6] == 'device':
            device_list.append(i)
        else:
            controller_list.append(i)
            
    print("\n\n")
    print(device_list, '\n')
    print(controller_list)


    device_count = len(device_list)
    print(device_count)
    for s in device_list:
        node_description = {}
        node_description['name'] = re.findall("name=(\w+)", s ) #separate
        # node_description['load_profile'] = None
        # node_description['devices'] = []
        device = {}

        device['custom_device_configs'] = {}

     
        device['node'] = re.findall(r"node=(\w+)", s) #separate
        device['class'] = re.findall(r"class=(\w+)", s) #separate
        device['custom_device_configs']['control_mode'] = re.findall(r"control_mode=(\w+)", s)
        device['node_default_control_setting'] = re.findall(r"default_control_setting=(\w+)", s)
        device['custom_device_configs']['total_energy_capacity'] =  re.findall(r"total_energy_capcity=(\w+)", s)
        device['custom_device_configs']['starting_energy'] =  re.findall(r"starting_energy=(\w+)", s)

        device['custom_device_configs']['minimum_energy_allowable'] =  re.findall(r"minimum_energy_allowable=(\w+)", s)
        device['custom_device_configs']['maximum_energy_allowable'] = re.findall(r"maximum_energy_allowable=(\w+)", s)


        device['custom_device_configs']['starting_SOC'] = re.findall(r"starting_SOC=(\w+)", s)
        device['custom_device_configs']['minimum_SOC'] = re.findall(r"minimum_SOC=(\w+)", s)
        device['custom_device_configs']['maximum_SOC'] = re.findall(r"maximum_SOC=(\w+)", s)

        device['custom_device_configs']['minimum_power_input_charge'] = re.findall(r"minimum_power_input=(\w+)", s)
        device['custom_device_configs']['maximum_power_input_discharge'] = re.findall(r"maximum_power_input=(\w+)", s)

        device['custom_device_configs']['max_power_ramp_rate'] = re.findall(r"max_power_ramp_rate=(\w+)", s)

        device['custom_device_configs']['charge_efficiency'] =  re.findall(r"charge_efficiency=(\w+)", s)
        device['custom_device_configs']['discharge_efficiency'] = re.findall(r"discharge_efficiency=(\w+)", s)

        device['custom_device_configs']['low_pass_filter_freq'] = re.findall(r"low_pass_filter_freq=(\w+)", s)
        
        node_description['devices'].append(device)
        json_query['scenario_config']['nodes'].append(node_description)
        # print('name', name)
        # print('node', node)
        # print('class', bsd_class)
        # print('control_mode', control_mode)
        # print('def_control_setting', default_control_setting)
        # print('total_energy_capacity', total_energy_capacity)
        # print('starting_energy', starting_energy)
        # print('min allow', minimum_energy_allowable)
        # print('max allow', maximum_energy_allowable)
        # print('starting soc', starting_SOC)
        # print('min SOC', minimum_SOC)
        # print('max SOC', maximum_SOC)
        # print('min input charge', minimum_power_input_charge)
        # print('max input charge', maximum_power_input_discharge)
        # print('charge_efficiency', charge_efficiency)
        # print('discharge e', discharge_efficiency)
        # print('lpf w', low_pass_filter_freq)
        # print("\n")

   
    controller_count = len(controller_list)
    for controller in controller_list: 
        device = {}
        device['name'] = re.findall(r"name=(\w+)", controller)
        device['node'] = re.findall(r"node=(\w+)", controller)
        device['class'] = re.findall(r"class=(\w+)", controller)
        device['custom_controller_configs']['control_mode'] = re.findall(r"control_mode=(\w+)", controller)
        device['custom_controller_configs']['active_power_target'] = re.findall(r"active_power_target=(\w+)", controller)
        
        device['custom_controller_configs']['default_control_setting'] = re.findall(r"default_control_setting=(\w+)", s)
        device['custom_controller_configs']['eta'] = re.findall(r"eta=(\w+)", controller)
        device['custom_controller_configs']['lpf_freq'] = re.findall(r"low_pass_filter_freq=(\w+)", controller)
        node_description['devices'].append(device)
        json_query['scenario_config']['nodes'].append(node_description)

        # print('name', name)
        # print('node', node)
        # print('class', bps_class)
        # print('def_control_setting', default_control_setting)
        
        # print('control_mode', control_mode)
        # print('eta', eta)
        # print('active_power_target', active_power_target)
        # print('lpf w', lpf_freq)
    




    return json_query
