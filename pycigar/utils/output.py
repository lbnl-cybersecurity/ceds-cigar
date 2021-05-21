import os
import matplotlib
matplotlib.use('agg', force=True, warn=False)
import matplotlib.pyplot as plt
import numpy as np
from pycigar.envs.wrappers.wrappers_constants import ACTION_RANGE
from pycigar.utils.logging import logger
from matplotlib.ticker import FormatStrFormatter
import re

BASE_VOLTAGE = 120

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
    output_specs['Substation Power Factor (%)'] = (
        substation_p / np.sqrt(substation_p ** 2 + substation_q ** 2)
    ).tolist()
    output_specs['Consumption']['DG Output (W)'] = np.sum(
        np.array([log_dict[node]['p'] for node in node_ids]), axis=0
    ).tolist()
    output_specs['Substation Top Voltage(V)'] = np.array(log_dict['network']['substation_top_voltage']).tolist()
    output_specs['Substation Bottom Voltage(V)'] = np.array(log_dict['network']['substation_bottom_voltage']).tolist()

    for inverter_name in output_specs['Inverter Outputs'].keys():
        node_id = env.k.device.get_node_connected_to(inverter_name)
        output_specs['Inverter Outputs'][inverter_name]['Voltage (V)'] = (
            np.array(log_dict[node_id]['voltage']) * BASE_VOLTAGE
        ).tolist()
        output_specs['Inverter Outputs'][inverter_name]['Power Output (W)'] = np.array(
            log_dict[inverter_name]['p_out']
        ).tolist()
        output_specs['Inverter Outputs'][inverter_name]['Reactive Power Output (VAR)'] = np.array(
            log_dict[inverter_name]['q_out']
        ).tolist()

    for regulator_name in output_specs['Regulator_testReg'].keys():
        if regulator_name != 'RegPhases':
            output_specs['Regulator_testReg'][regulator_name] = np.array(
                log_dict[regulator_name]['tap_number']
            ).tolist()

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

def plot_adv(
    log_dict, custom_metrics, epoch='', unbalance=False, is_def_win='', num_def_win='', num_act='', train_policy=''
):

    plt.rc('font', size=15)
    plt.rc('figure', titlesize=35)

    inv_k = next(k for k in log_dict if 'inverter' in k)
    f, ax = plt.subplots(7, 2, figsize=(30, 40))
    title = '[epoch {}][time {}][hack {}][def_win {}][{}/{}][policy {}] Defense reward: {:.2f}, Attack reward: {:.2f}'.format(
        epoch,
        custom_metrics['start_time'],
        custom_metrics['hack'],
        is_def_win,
        num_def_win,
        num_act,
        train_policy,
        sum(log_dict[inv_k]['reward']),
        sum(log_dict['adversary_' + inv_k]['reward']),
    )
    f.suptitle(title)
    ax[0, 0].plot(log_dict[log_dict[inv_k]['node']]['voltage'], color='tab:blue', label='voltage')

    ax[1, 0].plot(log_dict[inv_k]['y'], color='tab:blue', label='oscillation observer')

    ax[2, 0].plot(log_dict[inv_k]['q_set'], color='tab:blue', label='q_set')
    ax[2, 0].plot(log_dict[inv_k]['q_out'], color='tab:orange', label='q_val')

    labels = ['a1', 'a2', 'a3', 'a4', 'a5']
    a1, a2, a3, a4, a5 = ax[3, 0].plot(log_dict[inv_k]['control_setting'])
    ax[3, 0].set_ylabel('action')
    ax[3, 0].grid(b=True, which='both')
    ax[3, 0].legend([a1, a2, a3, a4, a5], labels, loc=1)

    a1, a2, a3, a4, a5 = ax[4, 0].plot(log_dict['adversary_' + inv_k]['control_setting'])
    ax[4, 0].set_ylabel('action')
    ax[4, 0].grid(b=True, which='both')
    ax[4, 0].legend([a1, a2, a3, a4, a5], labels, loc=1)

    ax[5, 0].plot(log_dict[inv_k]['sbar_solarirr'], color='tab:blue', label='sbar solar irr')
    ax[6, 0].plot(log_dict[inv_k]['sbar_pset'], color='tab:blue', label='sbar pset')

    # ax[0].set_ylim([0.93, 1.07])
    # ax[1].set_ylim([0, 0.8])
    # ax[2].set_ylim([-280, 280])
    # ax[3].set_ylim([-0.06, 0.06])
    # ax[4].set_ylim([-0.06, 0.06])
    labels = ['a1', 'a2', 'a3', 'a4', 'a5']
    a1, a2, a3, a4, a5 = ax[0, 1].plot(log_dict[inv_k]['control_setting'])
    ax[0, 1].set_ylabel('action')
    ax[0, 1].grid(b=True, which='both')
    ax[0, 1].legend([a1, a2, a3, a4, a5], labels, loc=1)

    a1, a2, a3, a4, a5 = ax[1, 1].plot(log_dict['adversary_' + inv_k]['control_setting'])
    ax[1, 1].set_ylabel('action')
    ax[1, 1].grid(b=True, which='both')
    ax[1, 1].legend([a1, a2, a3, a4, a5], labels, loc=1)

    ax[2, 1].plot(log_dict['component_observation']['component_y'], label='obs_component_y')
    ax[3, 1].plot(log_dict['component_observation']['component_pset'], label='obs_component_pset')
    ax[4, 1].plot(log_dict['component_observation']['component_ymax'], label='obs_component_ymax')
    component_y = np.array(log_dict['component_reward']['component_y'])
    component_oa = np.array(log_dict['component_reward']['component_oa'])
    component_init = np.array(log_dict['component_reward']['component_init'])
    component_pset_pmax = np.array(log_dict['component_reward']['component_pset_pmax'])

    total_reward = component_y + component_oa + component_init + component_pset_pmax

    ax[5, 1].plot(-component_y, label='abs_reward_component_y')
    ax[5, 1].plot(-component_oa, label='abs_reward_component_oa')
    ax[5, 1].plot(-component_init, label='abs_reward_component_init')
    ax[5, 1].plot(-component_pset_pmax, label='abs_reward_component_pset_pmax')
    ax[5, 1].plot(-total_reward, label='abs_total_reward')

    # x = range(len(component_y))
    # y_stack = np.cumsum(np.array([component_y, component_oa, component_init, component_pset_pmax]), axis=0)   # a 3x10 array
    # y_stack = y_stack/total_reward
    # ax[6, 1].fill_between(x, 0, y_stack[0,:], facecolor="#CC6666", alpha=.7, label='reward_component_y')
    # ax[6, 1].fill_between(x, y_stack[0,:], y_stack[1,:], facecolor="#1DACD6", alpha=.7, label='reward_component_oa')
    # ax[6, 1].fill_between(x, y_stack[1,:], y_stack[2,:], facecolor="#6E5160", label='reward_component_init')
    # ax[6, 1].fill_between(x, y_stack[2,:], y_stack[3,:], facecolor="#E3F59C", label='reward_component_pset_pmax')
    adv_component_y = np.array(log_dict['adv_component_reward']['adv_component_y'])
    adv_component_oa = np.array(log_dict['adv_component_reward']['adv_component_oa'])

    total_reward = adv_component_y + adv_component_oa

    ax[6, 1].plot(adv_component_y, label='abs_adv_reward_component_y')
    ax[6, 1].plot(-adv_component_oa, label='abs_adv_reward_component_oa')
    # ax[5, 1].plot(-component_init, label='abs_reward_component_init')
    # ax[5, 1].plot(-component_pset_pmax, label='abs_reward_component_pset_pmax')
    # ax[5, 1].plot(-total_reward, label='abs_total_reward')

    for i in range(7):
        for j in range(2):
            ax[i, j].grid(b=True, which='both')
            ax[i, j].legend(loc=1, ncol=2)
    ax[6, 1].legend(loc=4, ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return f


def plot_cluster(log_dict, custom_metrics, epoch='', unbalance=False, multiagent=False):
    def get_translation_and_slope(a_val, init_a):
        points = np.array(a_val)
        slope = points[:, 1] - points[:, 0]
        translation = points[:, 2] - init_a[2]
        return translation, slope

    plt.rc('font', size=15)
    plt.rc('figure', titlesize=35)

    cluster = {1: ['s701a', 's701b', 's701c', 's712c', 's713c', 's714a', 's714b', 's718a', 's720c', 's722b', 's722c', 's724b', 's725b'],
     2: ['s727c', 's728', 's729a', 's730c', 's731b', 's732c', 's733a'],
     3: ['s734c', 's735c', 's736b', 's737a', 's738a', 's740c', 's741c', 's742a', 's742b', 's744a']
    }

    f, ax = plt.subplots(4, 3, figsize=(40, 30))
    title = '[epoch {}][time {}][hack {}] reward: agent_1: {}, agent_2: {}, agent_3: {}'.format(
            epoch, custom_metrics['start_time'], custom_metrics['hack'], sum(log_dict['1']['reward']), sum(log_dict['2']['reward']), sum(log_dict['3']['reward'])
        )
    f.suptitle(title)

    voltage = np.array(log_dict['inverter_s701a']['v'])
    ax[0, 0].plot(voltage[:, 0], label='voltage_a')
    ax[0, 0].plot(voltage[:, 1], label='voltage_b')
    ax[0, 0].plot(voltage[:, 2], label='voltage_c')
    ax[1, 0].plot(np.array(log_dict['inverter_s701a']['u'])*100, label='unbalance')
    ax[1, 0].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='unbalance_worst')
    ax[2, 0].plot(log_dict['inverter_s701a']['y'], label='oscillation')
    translation, slope = get_translation_and_slope(log_dict['inverter_s701a']['control_setting'], custom_metrics['init_control_settings']['inverter_s701a'])
    ax[3, 0].plot(translation, color='tab:blue', label='act_phase_a')
    translation, slope = get_translation_and_slope(log_dict['inverter_s701b']['control_setting'], custom_metrics['init_control_settings']['inverter_s701b'])
    ax[3, 0].plot(translation, color='tab:orange', label='act_phase_b')
    translation, slope = get_translation_and_slope(log_dict['inverter_s701c']['control_setting'], custom_metrics['init_control_settings']['inverter_s701c'])
    ax[3, 0].plot(translation, color='tab:green', label='act_phase_c')
    ax[0, 0].set_ylim([0.93, 1.07])
    ax[1, 0].set_ylim([0, 6.5])
    ax[1, 0].axhline(3, linestyle='--', color='orange')
    ax[1, 0].axhline(6, linestyle='--', color='red')
    ax[2, 0].set_ylim([0, 0.5])
    ax[3, 0].set_ylim([-0.1, 0.1])

    voltage = np.array(log_dict['inverter_s727c']['v'])
    ax[0, 1].plot(voltage[:, 0], label='voltage_a')
    ax[0, 1].plot(voltage[:, 1], label='voltage_b')
    ax[0, 1].plot(voltage[:, 2], label='voltage_c')
    ax[1, 1].plot(np.array(log_dict['inverter_s727c']['u'])*100, label='unbalance')
    ax[1, 1].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='unbalance_worst')
    ax[2, 1].plot(log_dict['inverter_s727c']['y'], label='oscillation')
    translation, slope = get_translation_and_slope(log_dict['inverter_s729a']['control_setting'], custom_metrics['init_control_settings']['inverter_s729a'])
    ax[3, 1].plot(translation, color='tab:blue', label='act_phase_a')
    translation, slope = get_translation_and_slope(log_dict['inverter_s731b']['control_setting'], custom_metrics['init_control_settings']['inverter_s731b'])
    ax[3, 1].plot(translation, color='tab:orange', label='act_phase_b')
    translation, slope = get_translation_and_slope(log_dict['inverter_s732c']['control_setting'], custom_metrics['init_control_settings']['inverter_s732c'])
    ax[3, 1].plot(translation, color='tab:green', label='act_phase_c')
    ax[0, 1].set_ylim([0.93, 1.07])
    ax[1, 1].set_ylim([0, 6.5])
    ax[1, 1].axhline(3, linestyle='--', color='orange')
    ax[1, 1].axhline(6, linestyle='--', color='red')
    ax[2, 1].set_ylim([0, 0.5])
    ax[3, 1].set_ylim([-0.1, 0.1])

    voltage = np.array(log_dict['inverter_s736b']['v'])
    ax[0, 2].plot(voltage[:, 0], label='voltage_a')
    ax[0, 2].plot(voltage[:, 1], label='voltage_b')
    ax[0, 2].plot(voltage[:, 2], label='voltage_c')
    ax[1, 2].plot(np.array(log_dict['inverter_s734c']['u'])*100, label='unbalance')
    ax[1, 2].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='unbalance_worst')
    ax[2, 2].plot(log_dict['inverter_s734c']['y'], label='oscillation')
    translation, slope = get_translation_and_slope(log_dict['inverter_s737a']['control_setting'], custom_metrics['init_control_settings']['inverter_s737a'])
    ax[3, 2].plot(translation, color='tab:blue', label='act_phase_a')
    translation, slope = get_translation_and_slope(log_dict['inverter_s736b']['control_setting'], custom_metrics['init_control_settings']['inverter_s736b'])
    ax[3, 2].plot(translation, color='tab:orange', label='act_phase_b')
    translation, slope = get_translation_and_slope(log_dict['inverter_s736b']['control_setting'], custom_metrics['init_control_settings']['inverter_s736b'])
    ax[3, 2].plot(translation, color='tab:green', label='act_phase_c')
    ax[0, 2].set_ylim([0.93, 1.07])
    ax[1, 2].set_ylim([0, 6.5])
    ax[1, 2].axhline(3, linestyle='--', color='orange')
    ax[1, 2].axhline(6, linestyle='--', color='red')
    ax[2, 2].set_ylim([0, 0.5])
    ax[3, 2].set_ylim([-0.1, 0.1])

    for row in ax:
        for a in row:
            a.grid(b=True, which='both')
            a.legend(loc=1, ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return f


def plot_new(log_dict, custom_metrics, epoch='', unbalance=False, multiagent=False):
    def get_translation_and_slope(a_val, init_a):
        points = np.array(a_val)
        slope = points[:, 1] - points[:, 0]
        translation = points[:, 2] - init_a[2]
        return translation, slope

    plt.rc('font', size=15)
    plt.rc('figure', titlesize=35)

    if not unbalance:
        invs = [k for k in log_dict if k.startswith('inverter_s49') or k.startswith('inverter_s701')]
        inv_k = invs[0]
        f, ax = plt.subplots(3, 2, figsize=(30, 20))
        title = '[epoch {}][time {}][hack {}] total reward: {:.2f}'.format(
            epoch, custom_metrics['start_time'], custom_metrics['hack'], sum(log_dict[inv_k]['reward'])
        )
        f.suptitle(title)
        for inv in invs:
            ax[0, 0].plot(log_dict[inv[9:]]['voltage'], label='voltage_{}'.format(inv[-1]))

        ax[1, 0].plot(log_dict['y_metrics']['y_worst'], label='y worst')
        ax[1, 0].plot(log_dict['y_metrics']['y_mean'], label='y mean')

        ax[2, 0].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='unbalance observer %')
        ax[2, 0].plot(np.array(log_dict['u_metrics']['u_mean'])*100, label='unbalance mean %')


        ax[0, 0].set_ylim([0.93, 1.07])
        ax[1, 0].set_ylim([0, 0.8])
        ax[2, 0].set_ylim([0, 6])
        ax[0, 1].set_ylim([-0.1, 0.1])
        ax[1, 1].set_ylim([-0.06, 0.06])

        for inv in log_dict:
            if 'adversary_' not in inv and 'inverter_' in inv:
                translation, slope = get_translation_and_slope(
                    log_dict[inv]['control_setting'], custom_metrics['init_control_settings'][inv]
                )
                if inv[-1] == 'a':
                    color = 'tab:blue'
                elif inv[-1] == 'b':
                    color = 'tab:orange'
                elif inv[-1] == 'c':
                    color = 'tab:green'
                else:
                    color = 'tab:red'
                ax[0, 1].plot(translation, color=color)    
                ax[0, 1].plot(slope, color='tab:purple')
            elif 'adversary_' in inv:
                translation, slope = get_translation_and_slope(
                    log_dict[inv]['control_setting'],
                    custom_metrics['init_control_settings'][inv],
                )
                if inv[-1] == 'a':
                    color = 'tab:blue'
                elif inv[-1] == 'b':
                    color = 'tab:orange'
                elif inv[-1] == 'c':
                    color = 'tab:green'
                else:
                    color = 'tab:red'
                ax[1, 1].plot(translation, color=color)
                ax[1, 1].plot(slope, color='tab:red')

        #ax[4, 1].plot(log_dict['component_observation']['component_ymax'], label='obs_component_ymax')
        component_y = np.array(log_dict['component_reward']['component_y'])
        component_oa = np.array(log_dict['component_reward']['component_oa'])
        component_init = np.array(log_dict['component_reward']['component_init'])
        component_pset_pmax = np.array(log_dict['component_reward']['component_pset_pmax'])

        total_reward = component_y + component_oa + component_init + component_pset_pmax

        regs = [i for i in log_dict.keys() if 'reg' in i]
        for reg in regs:
            ax[-1, 1].plot(log_dict[reg]['tap_number'], label=reg)
        for i in range(3):
            for j in range(2):
                ax[i, j].grid(b=True, which='both')
                ax[i, j].legend(loc=1, ncol=2)

    for row in ax:
        for a in row:
            a.grid(b=True, which='both')
            a.legend(loc=1, ncol=2)
            a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return f


def plot_cluster(log_dict, custom_metrics, epoch='', unbalance=False, multiagent=False):
    def get_translation_and_slope(a_val, init_a):
        points = np.array(a_val)
        slope = points[:, 1] - points[:, 0]
        translation = points[:, 2] - init_a[2]
        return translation, slope

    plt.rc('font', size=15)
    plt.rc('figure', titlesize=35)

    cluster = {1: ['s1a', 's2b', 's4c', 's5c', 's6c', 's7a', 's9a', 's10a', 's11a', 's12b', 's16c', 's17c', 's34c'],
               2: ['s52a', 's53a', 's55a', 's56b', 's58b', 's59b', 's86b', 's87b', 's88a', 's90b', 's92c', 's94a', 's95b', 's96b'],
               3: ['s19a', 's20a', 's22b', 's24c', 's28a', 's29a', 's30c', 's31c', 's32c', 's33a', 's35a', 's37a', 's38b', 's39b', 's41c', 's42a', 's43b', 's45a', 's46a', 's47', 's48', 's49a', 's49b', 's49c', 's50c', 's51a'],
               4: ['s102c', 's103c', 's104c', 's106b', 's107b', 's109a', 's111a', 's112a', 's113a', 's114a'],
               5: ['s60a', 's62c', 's63a', 's64b', 's65a', 's65b', 's65c', 's66c', 's68a', 's69a', 's70a', 's71a', 's73c', 's74c', 's75c', 's76a', 's76b', 's76c', 's77b', 's79a', 's80b', 's82a', 's83c', 's84c', 's85c', 's98a', 's99b', 's100c']}

    f, ax = plt.subplots(5, 5, figsize=(30, 15))
    # title = '[epoch {}][time {}][hack {}] reward: a_1: {:.2f}, a_2: {:.2f}, a_3: {:.2f}, a_4: {:.2f}, a_5: {:.2f}'.format(
    #         epoch, custom_metrics['start_time'], custom_metrics['hack'],
    #         sum(log_dict['1']['reward']), sum(log_dict['2']['reward']),
    #         sum(log_dict['3']['reward']), sum(log_dict['4']['reward']),
    #         sum(log_dict['5']['reward'])
    #     )
    title = '[epoch {}][time {}][hack {}]'.format(
            epoch, custom_metrics['start_time'], custom_metrics['hack']
        )
    f.suptitle(title)

    voltage = np.array(log_dict['inverter_s1a']['v'])
    ax[0, 0].plot(voltage[:, 0], label='voltage_a')
    ax[0, 0].plot(voltage[:, 1], label='voltage_b')
    ax[0, 0].plot(voltage[:, 2], label='voltage_c')
    ax[1, 0].plot(np.array(log_dict['inverter_s1a']['u'])*100, label='unbalance')
    ax[1, 0].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='unbalance_worst')
    ax[2, 0].plot(np.array(log_dict['y_metrics']['y_worst']), label='oscillation_worst')
    ax[2, 0].plot(log_dict['inverter_s1a']['y'], label='oscillation')
    translation, slope = get_translation_and_slope(log_dict['inverter_s1a']['control_setting'], custom_metrics['init_control_settings']['inverter_s1a'])
    ax[3, 0].plot(translation, color='tab:blue', label='act_phase_a')
    translation, slope = get_translation_and_slope(log_dict['inverter_s2b']['control_setting'], custom_metrics['init_control_settings']['inverter_s2b'])
    ax[3, 0].plot(translation, color='tab:orange', label='act_phase_b')
    translation, slope = get_translation_and_slope(log_dict['inverter_s4c']['control_setting'], custom_metrics['init_control_settings']['inverter_s4c'])
    ax[3, 0].plot(translation, color='tab:green', label='act_phase_c')
    ax[0, 0].set_ylim([0.93, 1.07])
    ax[1, 0].set_ylim([0, 6.5])
    ax[1, 0].axhline(3, linestyle='--', color='orange')
    ax[1, 0].axhline(6, linestyle='--', color='red')
    ax[2, 0].set_ylim([0, 0.5])
    ax[3, 0].set_ylim([-0.1, 0.1])

    voltage = np.array(log_dict['inverter_s52a']['v'])
    ax[0, 1].plot(voltage[:, 0], label='voltage_a')
    ax[0, 1].plot(voltage[:, 1], label='voltage_b')
    ax[0, 1].plot(voltage[:, 2], label='voltage_c')
    ax[1, 1].plot(np.array(log_dict['inverter_s52a']['u'])*100, label='unbalance')
    ax[1, 1].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='unbalance_worst')
    ax[2, 1].plot(np.array(log_dict['y_metrics']['y_worst']), label='oscillation_worst')
    ax[2, 1].plot(log_dict['inverter_s52a']['y'], label='oscillation')
    translation, slope = get_translation_and_slope(log_dict['inverter_s52a']['control_setting'], custom_metrics['init_control_settings']['inverter_s52a'])
    ax[3, 1].plot(translation, color='tab:blue', label='act_phase_a')
    translation, slope = get_translation_and_slope(log_dict['inverter_s56b']['control_setting'], custom_metrics['init_control_settings']['inverter_s56b'])
    ax[3, 1].plot(translation, color='tab:orange', label='act_phase_b')
    translation, slope = get_translation_and_slope(log_dict['inverter_s92c']['control_setting'], custom_metrics['init_control_settings']['inverter_s92c'])
    ax[3, 1].plot(translation, color='tab:green', label='act_phase_c')
    ax[0, 1].set_ylim([0.93, 1.07])
    ax[1, 1].set_ylim([0, 6.5])
    ax[1, 1].axhline(3, linestyle='--', color='orange')
    ax[1, 1].axhline(6, linestyle='--', color='red')
    ax[2, 1].set_ylim([0, 0.5])
    ax[3, 1].set_ylim([-0.1, 0.1])

    voltage = np.array(log_dict['inverter_s28a']['v'])
    ax[0, 2].plot(voltage[:, 0], label='voltage_a')
    ax[0, 2].plot(voltage[:, 1], label='voltage_b')
    ax[0, 2].plot(voltage[:, 2], label='voltage_c')
    ax[1, 2].plot(np.array(log_dict['inverter_s28a']['u'])*100, label='unbalance')
    ax[1, 2].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='unbalance_worst')
    ax[2, 2].plot(np.array(log_dict['y_metrics']['y_worst']), label='oscillation_worst')
    ax[2, 2].plot(log_dict['inverter_s28a']['y'], label='oscillation')
    translation, slope = get_translation_and_slope(log_dict['inverter_s19a']['control_setting'], custom_metrics['init_control_settings']['inverter_s19a'])
    ax[3, 2].plot(translation, color='tab:blue', label='act_phase_a')
    translation, slope = get_translation_and_slope(log_dict['inverter_s22b']['control_setting'], custom_metrics['init_control_settings']['inverter_s22b'])
    ax[3, 2].plot(translation, color='tab:orange', label='act_phase_b')
    translation, slope = get_translation_and_slope(log_dict['inverter_s24c']['control_setting'], custom_metrics['init_control_settings']['inverter_s24c'])
    ax[3, 2].plot(translation, color='tab:green', label='act_phase_c')
    ax[0, 2].set_ylim([0.93, 1.07])
    ax[1, 2].set_ylim([0, 6.5])
    ax[1, 2].axhline(3, linestyle='--', color='orange')
    ax[1, 2].axhline(6, linestyle='--', color='red')
    ax[2, 2].set_ylim([0, 0.5])
    ax[3, 2].set_ylim([-0.1, 0.1])

    voltage = np.array(log_dict['inverter_s102c']['v'])
    ax[0, 3].plot(voltage[:, 0], label='voltage_a')
    ax[0, 3].plot(voltage[:, 1], label='voltage_b')
    ax[0, 3].plot(voltage[:, 2], label='voltage_c')
    ax[1, 3].plot(np.array(log_dict['inverter_s102c']['u'])*100, label='unbalance')
    ax[1, 3].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='unbalance_worst')
    ax[2, 3].plot(np.array(log_dict['y_metrics']['y_worst']), label='oscillation_worst')
    ax[2, 3].plot(log_dict['inverter_s102c']['y'], label='oscillation')
    translation, slope = get_translation_and_slope(log_dict['inverter_s114a']['control_setting'], custom_metrics['init_control_settings']['inverter_s114a'])
    ax[3, 3].plot(translation, color='tab:blue', label='act_phase_a')
    translation, slope = get_translation_and_slope(log_dict['inverter_s107b']['control_setting'], custom_metrics['init_control_settings']['inverter_s107b'])
    ax[3, 3].plot(translation, color='tab:orange', label='act_phase_b')
    translation, slope = get_translation_and_slope(log_dict['inverter_s104c']['control_setting'], custom_metrics['init_control_settings']['inverter_s104c'])
    ax[3, 3].plot(translation, color='tab:green', label='act_phase_c')
    ax[0, 3].set_ylim([0.93, 1.07])
    ax[1, 3].set_ylim([0, 6.5])
    ax[1, 3].axhline(3, linestyle='--', color='orange')
    ax[1, 3].axhline(6, linestyle='--', color='red')
    ax[2, 3].set_ylim([0, 0.5])
    ax[3, 3].set_ylim([-0.1, 0.1])

    voltage = np.array(log_dict['inverter_s60a']['v'])
    ax[0, 4].plot(voltage[:, 0], label='voltage_a')
    ax[0, 4].plot(voltage[:, 1], label='voltage_b')
    ax[0, 4].plot(voltage[:, 2], label='voltage_c')
    ax[1, 4].plot(np.array(log_dict['inverter_s60a']['u'])*100, label='unbalance')
    ax[1, 4].plot(np.array(log_dict['u_metrics']['u_worst'])*100, label='unbalance_worst')
    ax[2, 4].plot(np.array(log_dict['y_metrics']['y_worst']), label='oscillation_worst')
    ax[2, 4].plot(log_dict['inverter_s60a']['y'], label='oscillation')
    translation, slope = get_translation_and_slope(log_dict['inverter_s60a']['control_setting'], custom_metrics['init_control_settings']['inverter_s60a'])
    ax[3, 4].plot(translation, color='tab:blue', label='act_phase_a')
    translation, slope = get_translation_and_slope(log_dict['inverter_s64b']['control_setting'], custom_metrics['init_control_settings']['inverter_s64b'])
    ax[3, 4].plot(translation, color='tab:orange', label='act_phase_b')
    translation, slope = get_translation_and_slope(log_dict['inverter_s62c']['control_setting'], custom_metrics['init_control_settings']['inverter_s62c'])
    ax[3, 4].plot(translation, color='tab:green', label='act_phase_c')
    ax[0, 4].set_ylim([0.93, 1.07])
    ax[1, 4].set_ylim([0, 6.5])
    ax[1, 4].axhline(3, linestyle='--', color='orange')
    ax[1, 4].axhline(6, linestyle='--', color='red')
    ax[2, 4].set_ylim([0, 0.5])
    ax[3, 4].set_ylim([-0.1, 0.1])

    reward_agents = [ra for ra in log_dict.keys() if 'reward_' in ra]
    for i, ra in enumerate(reward_agents):
        component_y = np.array(log_dict[ra]['M'])
        component_u = np.array(log_dict[ra]['T'])
        component_a = np.array(log_dict[ra]['N'])
        component_i = np.array(log_dict[ra]['P'])
        component_p = np.array(log_dict[ra]['Q'])

        ax[4, i+1].plot(-component_y, label='y')
        ax[4, i+1].plot(-component_u, label='u')
        ax[4, i+1].plot(-component_a, label='a')
        ax[4, i+1].plot(-component_i, label='i')
        ax[4, i+1].plot(-component_p, label='p')
        #ax[4, i+1].plot(-(component_y+component_u+component_a+component_i+component_p), label='abs_total')

    regs = [k for k in log_dict if 'reg' in k]
    for reg in regs:
        ax[4, 0].plot(log_dict[reg]['tap_number'])

    ax[4, 0].set_xlabel('time (s)')
    ax[4, 0].set_ylabel('tap number')

    for row in ax:
        for a in row:
            a.grid(b=True, which='both')
            a.legend(loc=1, ncol=1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return f


