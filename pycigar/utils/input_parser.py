import os
import pandas as pd
import numpy as np


def input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path=None, benchmark=False, percentage_hack=0.45, adv=False, norl_mode=False, vectorized_mode=False):
    """Take multiple .csv files and parse them into the .yml file that required by pycigar.
    Parameters
    ----------
    misc_inputs_path : str
        directory to miscellaneous settings for the experiment. Example can be found at ./data/ieee37busdata/misc_inputs.csv
    dss_path : str
        directory to .dss file. Example can be found at ./data/ieee37busdata/ieee37.dss
    load_solar_path : str
        directory to load and solar profiles for different inverters. Example can be found at ./data/ieee37busdata/load_solar_data.csv
    breakpoints_path : str, optional
        directory to default settings of different inverters. Defaults to None to use the default settings in this function, by default None
    benchmark : bool, optional
        whether the experiment is in benchmark mode. If true, disable the randomization at inverters filter. Defaults to False, by default False
    percentage_hack : float, optional
        percentage hack for all devices. Defaults to 0.45. Only have meaning when benchmark is True, by default 0.45
    adv : bool, optional
        whether the experiment is adversarial training. Defaults to False. If True, set the advesarial devices to use RL controllers, by default False
    Returns
    -------
    dict
        a dictionary contains full information to run the experiment
    """

    file_misc_inputs_path = misc_inputs_path
    file_dss_path = dss_path
    file_load_solar_path = load_solar_path
    file_breakpoints_path = breakpoints_path

    json_query = {
        'M': 50,  # weight for y-value in reward function
        'N': 10,  # weight for taking different action from the initial action
        'P': 10,  # weight for taking different action from last timestep action
        'Q': 0.5,
        'is_disable_y': True,
        'vectorized_mode': False,

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
    misc_inputs_data = pd.read_csv(file_misc_inputs_path, header=None)
    misc_inputs_data = misc_inputs_data.T
    new_header = misc_inputs_data.iloc[0]  # grab the first row for the header
    misc_inputs_data = misc_inputs_data[1:]  # take the data less the header row
    misc_inputs_data.columns = new_header  # set the header row as the df header
    misc_inputs_data = misc_inputs_data.to_dict()

    M = misc_inputs_data['Oscillation Penalty'][1]
    N = misc_inputs_data['Action Penalty'][1]
    P = misc_inputs_data['Deviation from Optimal Penalty'][1]
    Q = misc_inputs_data['PsetPmax Penalty'][1]
    power_factor = misc_inputs_data['power factor'][1]
    load_scaling_factor = misc_inputs_data['load scaling factor'][1]
    solar_scaling_factor = misc_inputs_data['solar scaling factor'][1]
    p_ramp_rate = misc_inputs_data['p ramp rate'][1]
    q_ramp_rate = misc_inputs_data['q ramp rate'][1]

    json_query['M'] = M
    json_query['N'] = N
    json_query['P'] = P
    json_query['Q'] = Q
    json_query['vectorized_mode'] = vectorized_mode
    json_query['scenario_config']['custom_configs']['load_scaling_factor'] = load_scaling_factor
    json_query['scenario_config']['custom_configs']['solar_scaling_factor'] = solar_scaling_factor
    json_query['scenario_config']['custom_configs']['power_factor'] = power_factor

    low_pass_filter_measure_mean = misc_inputs_data['measurement filter time constant mean'][1]
    low_pass_filter_measure_std = misc_inputs_data['measurement filter time constant std'][1]
    low_pass_filter_output_mean = misc_inputs_data['output filter time constant mean'][1]
    low_pass_filter_output_std = misc_inputs_data['output filter time constant std'][1]
    default_control_setting = [
        misc_inputs_data['bp1 default'][1],
        misc_inputs_data['bp2 default'][1],
        misc_inputs_data['bp3 default'][1],
        misc_inputs_data['bp4 default'][1],
        misc_inputs_data['bp5 default'][1],
    ]

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
        device['device'] = 'pv_device'
        device['custom_device_configs'] = {}
        device['custom_device_configs']['default_control_setting'] = node_default_control_setting
        device['custom_device_configs']['gain'] = 1e5
        device['custom_device_configs']['is_butterworth_filter'] = False
        device['custom_device_configs']['p_ramp_rate'] = p_ramp_rate
        device['custom_device_configs']['q_ramp_rate'] = q_ramp_rate
        device['custom_device_configs']['low_pass_filter_measure_mean'] = low_pass_filter_measure_mean
        device['custom_device_configs']['low_pass_filter_output_mean'] = low_pass_filter_output_mean
        if not benchmark:
            device['custom_device_configs']['low_pass_filter_measure_std'] = low_pass_filter_measure_std
            device['custom_device_configs']['low_pass_filter_output_std'] = low_pass_filter_output_std

        device['controller'] = 'rl_controller'
        device['custom_controller_configs'] = {}
        device['custom_controller_configs']['default_control_setting'] = node_default_control_setting
        if norl_mode:
            device['controller'] = 'adaptive_inverter_controller'
            device['custom_controller_configs']['delay_timer'] = 60
            device['custom_controller_configs']['threshold'] = 0.05
            device['custom_controller_configs']['adaptive_gain'] = 20

        device['adversary_controller'] = 'adaptive_fixed_controller'
        if adv:
            device['adversary_controller'] = 'rl_controller'
        device['adversary_custom_controller_configs'] = {}
        device['adversary_custom_controller_configs']['default_control_setting'] = [1.014, 1.015, 1.015, 1.016, 1.017]
        device['hack'] = [250, percentage_hack, 500]

        node_description['devices'].append(device)
        json_query['scenario_config']['nodes'].append(node_description)

    max_tap_change = misc_inputs_data['max tap change default'][1]
    forward_band = misc_inputs_data['forward band default'][1]
    tap_number = misc_inputs_data['tap number default'][1]
    tap_delay = misc_inputs_data['tap delay default'][1]
    delay = misc_inputs_data['delay default'][1]

    json_query['scenario_config']['regulators']['max_tap_change'] = max_tap_change
    json_query['scenario_config']['regulators']['forward_band'] = forward_band
    json_query['scenario_config']['regulators']['tap_number'] = tap_number
    json_query['scenario_config']['regulators']['tap_delay'] = tap_delay
    json_query['scenario_config']['regulators']['delay'] = delay

    return json_query
