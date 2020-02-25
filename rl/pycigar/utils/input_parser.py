import json 
import os
import pycigar.config as config 
import pandas as pd 
import numpy as np 

def input_parser(folder_name):
    folder_name = folder_name
    folder_path = os.path.join(config.DATA_DIR, folder_name)

    file_dss_name = 'ieee37.dss'
    file_dss_path = os.path.join(folder_path, file_dss_name)

    file_load_solar_name = 'load_solar_data.csv'
    file_load_solar_path = os.path.join(folder_path, file_load_solar_name)

    file_misc_inputs_name = 'misc_inputs.csv'
    file_misc_inputs_path = os.path.join(folder_path, file_misc_inputs_name)

    file_breakpoints_name = 'breakpoints.csv'
    file_breakpoints_path = os.path.join(folder_path, file_breakpoints_name)

    json_query = {
        'M': 50,  # weight for y-value in reward function
        'N': 10,  # weight for taking different action from the initial action
        'P': 10,  # weight for taking different action from last timestep action

        'tune_search': False,
        'hack_setting': {'default_control_setting': [1.039, 1.04, 1.04, 1.041, 1.042]},

        'env_config': {
          'clip_actions': True,
          'sims_per_step': 20
        },
        'simulation_config': {
          'network_model_directory': file_dss_path,
          'custom_configs':  {'solution_mode': 1,
                              'solution_number': 1,
                              'solution_step_size': 1,
                              'solution_control_mode': -1,
                              'solution_max_control_iterations': 1000000,
                              'solution_max_iterations': 30000,
                              'power_factor': 0.9},
        },
        'scenario_config': {
          'multi_config': True,
          'start_end_time': 500,
          'network_data_directory': file_load_solar_path,
          'custom_configs': {'load_scaling_factor': 1.5,
                             'solar_scaling_factor': 3,
                             'slack_bus_voltage': 1.04,
                             'load_generation_noise': False,
                             'power_factor': 0.9},
          'nodes': []
        }
    }

    # read misc_input
    misc_inputs_data = pd.read_csv(file_misc_inputs_path, header=None)
    misc_inputs_data = misc_inputs_data.T
    new_header = misc_inputs_data.iloc[0] #grab the first row for the header
    misc_inputs_data = misc_inputs_data[1:] #take the data less the header row
    misc_inputs_data.columns = new_header #set the header row as the df header
    misc_inputs_data = misc_inputs_data.to_dict()

    M = misc_inputs_data['Oscillation Penalty'][1]
    N = misc_inputs_data['Action Penalty'][1]
    P = misc_inputs_data['Deviation from Optimal Penalty'][1]
    power_factor = misc_inputs_data['power_factor'][1]
    load_scaling_factor = misc_inputs_data['load scaling factor'][1]
    solar_scaling_factor = misc_inputs_data['solar scaling factor'][1]
    
    json_query['M'] =  M
    json_query['N'] = N
    json_query['P'] = P
    json_query['scenario_config']['custom_configs']['load_scaling_factor'] = load_scaling_factor
    json_query['scenario_config']['custom_configs']['solar_scaling_factor'] = solar_scaling_factor
    json_query['scenario_config']['custom_configs']['power_factor'] = power_factor

    low_pass_filter_measure_mean = misc_inputs_data['measurement filter time constant mean'][1]
    low_pass_filter_measure_std = misc_inputs_data['measurement filter time constant std'][1]
    low_pass_filter_output_mean = misc_inputs_data['output filter time constant mean'][1]
    low_pass_filter_output_std = misc_inputs_data['output filter time constant std'][1]
    default_control_setting = [misc_inputs_data['bp1 default'][1],
                               misc_inputs_data['bp2 default'][1],
                               misc_inputs_data['bp3 default'][1],
                               misc_inputs_data['bp4 default'][1],
                               misc_inputs_data['bp5 default'][1]]

    # read load_solar_data & read 
    load_solar_data = pd.read_csv(file_load_solar_path)
    node_names = [node for node in list(load_solar_data) if '_pv' not in node]
    breakpoints_data = pd.read_csv(file_breakpoints_path)

    for node in node_names:
        node_default_control_setting = default_control_setting
        if node + '_pv' in list(breakpoints_data):
            node_default_control_setting = breakpoints_data[node + '_pv'].tolist()

        node_description = {}
        node_description['name'] = node.lower()
        node_description['load_profile'] = None
        node_description['devices'] = []
        device = {}
        device['name'] = 'inverter_' + node.lower()
        device['type'] = 'pv_device'
        device['controller'] = 'rl_controller'
        device['custom_configs'] = {}
        device['custom_configs']['default_control_setting'] = node_default_control_setting
        device['custom_configs']['delay_timer'] = 60
        device['custom_configs']['threshold'] = 0.05
        device['custom_configs']['adaptive_gain'] = 20
        device['custom_configs']['low_pass_filter_measure'] = low_pass_filter_measure_std*np.random.randn() + low_pass_filter_measure_mean
        device['custom_configs']['low_pass_filter_output'] = low_pass_filter_output_std*np.random.randn() + low_pass_filter_output_mean
        device['adversary_controller'] = 'fixed_controller'
        device['adversary_custom_configs'] = {}
        device['adversary_custom_configs']['default_control_setting'] = [1.014, 1.015, 1.015, 1.016, 1.017]
        device['hack'] = [300, 0.4]
        node_description['devices'].append(device)

        json_query['scenario_config']['nodes'].append(node_description)

    return json_query
