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


    #device_count = len(device_list)
    req = ['name', 'node', 'class']

    for s in device_list:
        node_description = {}
        node_description['devices'] = []
        device = {}
        device['custom_device_configs'] = {}

        key = re.findall(r"(\w+)=", s)
        val = re.findall(r"=(\w+)|=\[([\w,]+)\]", s)

        for v in range(len(val)):
            print(val[v])
            if val[v][0] != '':
                val[v] = val[v][0]
            else:          
                val[v] = val[v][1]

        for k in range(len(key)):
            if key[k] not in req:
                device['custom_device_configs'][key[k]] = val[k]
            else:
                node_description[key[k]] = val[k]
 
        node_description['devices'].append(device)
        json_query['scenario_config']['nodes'].append(node_description)
    

    #controller_count = len(controller_list)
    
    for controller in controller_list: 
        node_description = {}
        node_description['devices'] = []
        device = {}
        device['custom_controller_configs'] = {}
        key = re.findall(r"(\w+)=", controller )
        val = re.findall(r"=(\w+)|=\[([\w,]+)\]", controller )

        for v in range(len(val)):
            if val[v][0] != '':
                val[v] = val[v][0]
            else:   
                val[v] = val[v][1]
       
        for k in range(len(key)):
            if key[k] not in req:
                device['custom_controller_configs'][key[k]] = val[k]
            else:
                node_description[key[k]] = val[k]

        node_description['devices'].append(device)
        json_query['scenario_config']['nodes'].append(node_description)

    return json_query
