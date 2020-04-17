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


def plot_new(log_dict, epoch='', unbalance=False):
    def get_translation_and_slope(a_val):
        points = np.array(a_val)
        slope = points[:, 1] - points[:, 0]
        og_point = points[0, 2]
        translation = points[:, 2] - og_point
        return translation, slope

    plt.rc('font', size=15)
    plt.rc('figure', titlesize=35)

    if not unbalance:
        inv_k = next(k for k in log_dict if 'inverter' in k)
        f, ax = plt.subplots(5, figsize=(25, 20))
        title = '[epoch {}] total reward: {:.2f}'.format(epoch, sum(log_dict[inv_k]['reward']))
        f.suptitle(title)
        ax[0].plot(log_dict[log_dict[inv_k]['node']]['voltage'], color='tab:blue', label='voltage')

        ax[1].plot(log_dict[inv_k]['y'], color='tab:blue', label='oscillation observer')

        ax[2].plot(log_dict[inv_k]['q_set'], color='tab:blue', label='q_set')
        ax[2].plot(log_dict[inv_k]['q_out'], color='tab:orange', label='q_val')

        translation, slope = get_translation_and_slope(log_dict[inv_k]['control_setting'])
        ax[3].plot(translation, color='tab:blue', label='RL translation')
        ax[3].plot(slope, color='tab:purple', label='RL slope (a2-a1)')

        translation, slope = get_translation_and_slope(log_dict['adversary_' + inv_k]['control_setting'])
        ax[4].plot(translation, color='tab:orange', label='hacked translation')
        ax[4].plot(slope, color='tab:red', label='hacked slope (a2-a1)')
        ax[0].set_ylim([0.93, 1.07])
        ax[1].set_ylim([0, 0.8])
        ax[2].set_ylim([-280, 280])
        ax[3].set_ylim([-0.06, 0.06])
        ax[4].set_ylim([-0.06, 0.06])
    else:
        inv_ks = [k for k in log_dict if k.startswith('inverter_s701')]
        regs = [k for k in log_dict if 'reg' in k]
        reg = regs[0] if regs else None

        f, ax = plt.subplots(2 + len(inv_ks) + (reg is not None), figsize=(25, 8 + 4 * len(inv_ks) + 4 * (reg is not None)))
        title = '[epoch {}] total reward: {:.2f}'.format(epoch, sum(log_dict[inv_ks[0]]['reward']))
        f.suptitle(title)
        for i, k in enumerate(inv_ks):
            ax[0].plot(log_dict[log_dict[k]['node']]['voltage'], label='voltage ({})'.format(k))
            ax[1].plot(log_dict[k]['u'], label='unbalance observer ({})'.format(k))

            translation, slope = get_translation_and_slope(log_dict[k]['control_setting'])
            ax[2+i].plot(translation, label='RL translation ({})'.format(k))
            ax[2+i].plot(slope, label='RL slope (a2-a1) ({})'.format(k))

        if reg:
            ax[-1].plot(log_dict[reg]['tap_number'], label=reg)

        ax[0].set_ylim([0.90, 1.10])
        ax[1].set_ylim([0, 0.1])
        ax[2].set_ylim([-280, 280])
        for i in range(len(inv_ks)):
            ax[2+i].set_ylim([-0.06, 0.06])

    for a in ax:
        a.grid(b=True, which='both')
        a.legend(loc=1, ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return f
