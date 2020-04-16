from pycigar.utils.logging import logger
import matplotlib.pyplot as plt
import os
import numpy as np

BASE_VOLTAGE = 120


def plot(env, tracking_id, save_dir=None, file_name=None):
    log_dict = logger().log_dict
    f, ax = plt.subplots(6, figsize=(25, 25))
    tracking_id = tracking_id
    node = env.k.device.get_node_connected_to(tracking_id)
    ax[0].set_title(tracking_id + " -- total reward: " + str(0))
    ax[0].plot(log_dict[node]['voltage'])
    ax[0].set_ylabel('voltage')
    ax[0].set_ylim((0.925, 1.07))
    ax[0].grid(b=True, which='both')
    ax[1].plot(log_dict[tracking_id]['y'])
    ax[1].set_ylabel('oscillation observer')
    ax[1].grid(b=True, which='both')
    ax[2].plot(log_dict[tracking_id]['q_set'])
    ax[2].plot(log_dict[tracking_id]['q_out'])
    ax[2].set_ylabel('reactive power')
    ax[2].grid(b=True, which='both')
    labels = ['a1', 'a2', 'a3', 'a4', 'a5']
    [a1, a2, a3, a4, a5] = ax[3].plot(log_dict[tracking_id]['control_setting'])
    print('a: {}'.format(len(log_dict[tracking_id]['control_setting'])))
    ax[3].set_ylabel('action')
    ax[3].grid(b=True, which='both')
    plt.legend([a1, a2, a3, a4, a5], labels, loc=1)

    if save_dir and file_name:
        f.savefig(os.path.join(save_dir, file_name))
    plt.close(f)
    return f


def plot_unbalance(env, tracking_id, save_dir, file_name):
    log_dict = logger().log_dict
    f, ax = plt.subplots(6, figsize=(25, 25))
    tracking_id = tracking_id
    node = env.k.device.get_node_connected_to(tracking_id)
    plot_v_list = []
    plot_v_label = []
    v = ax[0].plot(log_dict[node]['voltage'])
    plot_v_list.append(v[0])
    if tracking_id[-1].isdigit():
        plot_v_label.append('all')
    elif tracking_id[-1] == 'a':
        plot_v_label.append('a')
    elif tracking_id[-1] == 'b':
        plot_v_label.append('b')
    elif tracking_id[-1] == 'c':
        plot_v_label.append('c')

    if tracking_id[:-1] + 'a' in log_dict.keys() and tracking_id[:-1] + 'a' != tracking_id:
        v = ax[0].plot(log_dict[tracking_id[:-1] + 'a'])
        plot_v_list.append(v[0])
        plot_v_label.append('a')
    if tracking_id[:-1] + 'b' in log_dict.keys() and tracking_id[:-1] + 'b' != tracking_id:
        v = ax[0].plot(log_dict[tracking_id[:-1] + 'b'])
        plot_v_list.append(v[0])
        plot_v_label.append('b')
    if tracking_id[:-1] + 'c' in log_dict.keys() and tracking_id[:-1] + 'c' != tracking_id:
        v = ax[0].plot(log_dict[tracking_id[:-1] + 'c'])
        plot_v_list.append(v[0])
        plot_v_label.append('c')

    ax[0].legend(plot_v_list, plot_v_label, loc=1)
    ax[0].set_ylabel('voltage')
    ax[0].grid(b=True, which='both')
    ax[1].plot(log_dict[tracking_id]['y'])
    ax[1].set_ylabel('oscillation observer')
    ax[1].grid(b=True, which='both')
    ax[2].plot(log_dict[tracking_id]['q_set'])
    ax[2].plot(log_dict[tracking_id]['q_out'])
    ax[2].set_ylabel('reactive power')
    ax[2].grid(b=True, which='both')
    labels = ['a1', 'a2', 'a3', 'a4', 'a5']
    [a1, a2, a3, a4, a5] = ax[3].plot(log_dict[tracking_id]['control_setting'])
    ax[3].set_ylabel('action')
    ax[3].grid(b=True, which='both')
    ax[3].legend([a1, a2, a3, a4, a5], labels, loc=1)

    tracking_id = 'reg1'
    ax[4].plot(log_dict[tracking_id]['control_setting'])
    ax[4].set_ylabel('reg_val' + tracking_id)

    if save_dir and file_name:
        f.savefig(os.path.join(save_dir, file_name))
    plt.close(f)
    return f


def pycigar_output_specs(env):
    log_dict = logger().log_dict
    output_specs = {}
    if isinstance(env.k.sim_params['scenario_config']['start_end_time'], list):
        start_end_time = env.k.sim_params['scenario_config']['start_end_time']
        output_specs['Start Time'] = start_end_time[0] + 50
        output_specs['Time Steps'] = start_end_time[1] - start_end_time[0]

    output_specs['Time Step Size (s)'] = 1  # TODO: current resolution

    output_specs['allMeterVoltages'] = {}
    output_specs['allMeterVoltages']['Min'] = []
    output_specs['allMeterVoltages']['Mean'] = []
    output_specs['allMeterVoltages']['StdDev'] = []
    output_specs['allMeterVoltages']['Max'] = []
    output_specs['Consumption'] = {}
    output_specs['Consumption']['Power Substation (W)'] = []
    output_specs['Consumption']['Losses Total (W)'] = []
    output_specs['Consumption']['DG Output (W)'] = []
    output_specs['Substation Power Factor (%)'] = []
    output_specs['Regulator_testReg'] = {}
    reg_phases = ''
    for regulator_name in env.k.device.get_regulator_device_ids():
        output_specs['Regulator_testReg'][regulator_name] = []
        reg_phases += regulator_name[-1]
    output_specs['Regulator_testReg']['RegPhases'] = reg_phases.upper()

    output_specs['Substation Top Voltage(V)'] = []
    output_specs['Substation Bottom Voltage(V)'] = []
    output_specs['Substation Regulator Minimum Voltage(V)'] = []
    output_specs['Substation Regulator Maximum Voltage(V)'] = []

    output_specs['Inverter Outputs'] = {}
    for inverter_name in env.k.device.get_pv_device_ids():
        output_specs['Inverter Outputs'][inverter_name] = {}
        output_specs['Inverter Outputs'][inverter_name]['Name'] = inverter_name
        output_specs['Inverter Outputs'][inverter_name]['Voltage (V)'] = []
        output_specs['Inverter Outputs'][inverter_name]['Power Output (W)'] = []
        output_specs['Inverter Outputs'][inverter_name]['Reactive Power Output (VAR)'] = []

    node_ids = env.k.node.get_node_ids()
    voltages = np.array([log_dict[node]['voltage'] for node in node_ids])

    output_specs['allMeterVoltages']['Min'] = (np.amin(voltages, axis=0) * BASE_VOLTAGE).tolist()
    output_specs['allMeterVoltages']['Max'] = (np.amax(voltages, axis=0) * BASE_VOLTAGE).tolist()
    output_specs['allMeterVoltages']['Mean'] = (np.mean(voltages, axis=0) * BASE_VOLTAGE).tolist()
    output_specs['allMeterVoltages']['StdDev'] = (np.std(voltages, axis=0) * BASE_VOLTAGE).tolist()

    substation_p = np.array(log_dict['network']['substation_power'])[:, 0]
    substation_q = np.array(log_dict['network']['substation_power'])[:, 1]
    loss_p = np.array(log_dict['network']['loss'])[:, 0]
    output_specs['Consumption']['Power Substation (W)'] = substation_p.tolist()
    output_specs['Consumption']['Losses Total (W)'] = loss_p.tolist()
    output_specs['Substation Power Factor (%)'] = (substation_p / np.sqrt(substation_p**2 + substation_q**2)).tolist()
    output_specs['Consumption']['DG Output (W)'] = np.sum(np.array([log_dict[node]['p'] for node in node_ids]), axis=0).tolist()
    output_specs['Substation Top Voltage(V)'] = np.array(log_dict['network']['substation_top_voltage']).tolist()
    output_specs['Substation Bottom Voltage(V)'] = np.array(log_dict['network']['substation_bottom_voltage']).tolist()

    for inverter_name in output_specs['Inverter Outputs'].keys():
        node_id = env.k.device.get_node_connected_to(inverter_name)
        output_specs['Inverter Outputs'][inverter_name]['Voltage (V)'] = (np.array(log_dict[node_id]['voltage']) * BASE_VOLTAGE).tolist()
        output_specs['Inverter Outputs'][inverter_name]['Power Output (W)'] = np.array(log_dict[inverter_name]['p_out']).tolist()
        output_specs['Inverter Outputs'][inverter_name]['Reactive Power Output (VAR)'] = np.array(log_dict[inverter_name]['q_out']).tolist()

    for regulator_name in output_specs['Regulator_testReg'].keys():
        if regulator_name != 'RegPhases':
            output_specs['Regulator_testReg'][regulator_name] = np.array(log_dict[regulator_name]['tap_number']).tolist()

        val_max = None
        val_min = None
        for regulator_name in output_specs['Regulator_testReg'].keys():
            if regulator_name != 'RegPhases':
                vreg = np.array(log_dict[regulator_name]['regulator_forwardvreg'])
                band = np.array(log_dict[regulator_name]['forward_band'])
                val_upper = vreg + band
                val_lower = vreg - band
                val_max = val_upper if val_max is None else np.amax([val_max, val_upper], axis=0)
                val_min = val_lower if val_min is None else np.amax([val_min, val_lower], axis=0)

        output_specs['Substation Regulator Minimum Voltage(V)'] = val_min.tolist()
        output_specs['Substation Regulator Maximum Voltage(V)'] = val_max.tolist()

    inverter_outputs = []
    for inverter_name in output_specs['Inverter Outputs'].keys():
        inverter_outputs.append(output_specs['Inverter Outputs'][inverter_name])
    output_specs['Inverter Outputs'] = inverter_outputs

    return output_specs
