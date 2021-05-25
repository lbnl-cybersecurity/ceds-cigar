import os
import pandas as pd
import numpy as np


def input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path=None, benchmark=False, percentage_hack=0.45, adv=False, norl_mode=False, vectorized_mode=False, battery_path=None, use_load_generator=False):
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
    def _battery_input_parser(battery_path):
        f = open(battery_path, "r")
        input_file = f.read()

        input_list = [x for x in input_file.split("\n")]

        device_list = []
        controller_list = []

        for i in input_list:
            i = i.split(' ')
            if i[0] == 'device':
                device_list.append(i[1:])
            elif i[0] == 'controller':
                controller_list.append(i[1:])

        for s in device_list:
            custom_props = {}
            for prop in s:
                key, val  = prop.split('=')
                try:
                    val = float(val)
                except ValueError:
                    pass

                if key == 'name':
                    name = val
                elif key == 'node':
                    node = val
                elif key == 'class':
                    device = val
                else:
                    custom_props[key] = val
            for n in json_query['scenario_config']['nodes']:
                if n['name'] == node:
                    n['devices'].append({'name': name,
                                         'device': device,
                                         'custom_device_configs': custom_props})

        if controller_list:
            json_query['scenario_config']['controllers'] = []

        for s in controller_list:
            custom_props = {}
            for prop in s:
                key, val = prop.split('=')
                try:
                    val = float(val)
                except ValueError:
                    pass

                if key == 'name':
                    name = val
                elif key == 'class':
                    controller = val
                elif key == 'devices':
                    list_devices = val.strip('[]').split(',')
                else:
                    custom_props[key] = val

            json_query['scenario_config']['controllers'].append({'name': name,
                                                                 'controller': controller,
                                                                 'custom_controller_configs': custom_props,
                                                                 'list_devices': list_devices
                                                                })

    file_misc_inputs_path = misc_inputs_path
    file_dss_path = dss_path
    file_load_solar_path = load_solar_path
    file_breakpoints_path = breakpoints_path

    json_query = {
        'M': 50,  # weight for y-value in reward function
        'N': 10,  # weight for taking different action from the initial action
        'P': 10,  # weight for taking different action from last timestep action
        'Q': 0.5,
        'T': 100,
        'Z': 100,
        'Y': 0,
        'is_disable_log': False,
        'is_disable_y': False,
        'vectorized_mode': False,

        'hack_setting': {'default_control_setting': [1.039, 1.04, 1.04, 1.041, 1.042]},
        'env_config': {'clip_actions': True, 'sims_per_step': 30},
        'attack_randomization': {'generator': 'AttackGenerator'},
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
            'use_load_generator': use_load_generator,
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
    misc_inputs_data = pd.read_csv(file_misc_inputs_path, header=None, index_col=0)

    M = float(misc_inputs_data.loc['Oscillation Penalty'])
    N = float(misc_inputs_data.loc['Action Penalty'])
    P = float(misc_inputs_data.loc['Deviation from Optimal Penalty'])
    Q = float(misc_inputs_data.loc['PsetPmax Penalty'])
    power_factor = float(misc_inputs_data.loc['power factor'])
    load_scaling_factor = float(misc_inputs_data.loc['load scaling factor'])
    solar_scaling_factor = float(misc_inputs_data.loc['solar scaling factor'])

    # check protection line
    if 'protection line' in misc_inputs_data.index and 'protection threshold' in misc_inputs_data.index:
        protection_line = str(misc_inputs_data.loc['protection line'].values[0])
        protection_threshold = str(misc_inputs_data.loc['protection threshold'].values[0])
    else:
        protection_line = None
        protection_threshold = None

    if protection_line and protection_threshold:
        json_query['protection'] = {}
        protection_line = protection_line.split()
        protection_threshold = protection_threshold.split()
        json_query['protection']['line'] = protection_line
        json_query['protection']['threshold'] = np.array(protection_threshold, dtype=float)

    # check split phase
    if 'split phase' in misc_inputs_data.index:
        split_phase = int(misc_inputs_data.loc['split phase'])
    else:
        split_phase = 0
    if split_phase != 0:
        json_query['split_phase'] = True
    else:
        json_query['split_phase'] = False
    json_query['M'] = M
    json_query['N'] = N
    json_query['P'] = P
    json_query['Q'] = Q
    json_query['vectorized_mode'] = vectorized_mode
    json_query['scenario_config']['custom_configs']['load_scaling_factor'] = load_scaling_factor
    json_query['scenario_config']['custom_configs']['solar_scaling_factor'] = solar_scaling_factor
    json_query['scenario_config']['custom_configs']['power_factor'] = power_factor

    low_pass_filter_measure_mean = float(misc_inputs_data.loc['measurement filter time constant mean'])
    low_pass_filter_measure_std = float(misc_inputs_data.loc['measurement filter time constant std'])
    low_pass_filter_output_mean = float(misc_inputs_data.loc['output filter time constant mean'])
    low_pass_filter_output_std = float(misc_inputs_data.loc['output filter time constant std'])
    default_control_setting = [
        misc_inputs_data.loc['bp1 default'],
        misc_inputs_data.loc['bp2 default'],
        misc_inputs_data.loc['bp3 default'],
        misc_inputs_data.loc['bp4 default'],
        misc_inputs_data.loc['bp5 default'],
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

        device['adversary_controller'] = ['adaptive_fixed_controller', 'adaptive_unbalanced_fixed_controller']
        if adv:
            device['adversary_controller'] = 'rl_controller'
        device['adversary_custom_controller_configs'] = {}
        device['adversary_custom_controller_configs']['default_control_setting'] = [1.014, 1.015, 1.015, 1.016, 1.017]
        device['hack'] = [250, percentage_hack, 500]

        node_description['devices'].append(device)
        json_query['scenario_config']['nodes'].append(node_description)

    if 'max tap change default' in misc_inputs_data.index:
        max_tap_change = misc_inputs_data.loc['max tap change default']
    else:
        max_tap_change = None
    if 'forward band default' in misc_inputs_data.index:
        forward_band = misc_inputs_data.loc['forward band default']
    else:
        forward_band = None
    if 'tap number default' in misc_inputs_data.index:
        tap_number = misc_inputs_data.loc['tap number default']
    else:
        tap_number = None
    if 'tap delay default' in misc_inputs_data.index:
        tap_delay = misc_inputs_data.loc['tap delay default']
    else:
        tap_delay = None
    if 'delay default' in misc_inputs_data.index:
        delay = misc_inputs_data.loc['delay default']
    else:
        delay = None

    json_query['scenario_config']['regulators']['max_tap_change'] = max_tap_change
    json_query['scenario_config']['regulators']['forward_band'] = forward_band
    json_query['scenario_config']['regulators']['tap_number'] = tap_number
    json_query['scenario_config']['regulators']['tap_delay'] = tap_delay
    json_query['scenario_config']['regulators']['delay'] = delay

    if battery_path:
        _battery_input_parser(battery_path)

    return json_query
