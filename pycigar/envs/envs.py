from pycigar.envs.central_env import CentralEnv
from pycigar.envs.wrappers import *


class CentralControlPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = SingleRelativeInitDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env


class NewCentralControlPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = NewSingleRelativeInitDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env


class CentralControlPVInverterContinuousEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = SingleRelativeInitContinuousActionWrapper(env)
        env = CentralGlobalRewardWrapper(env)
        env = CentralLocalObservationWrapper(env)
        env = CentralFramestackObservationWrapper(env)
        self.env = env


class CentralControlPhaseSpecificPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = SingleRelativeInitPhaseSpecificDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env, unbalance=True)
        env = CentralLocalObservationWrapper(env, unbalance=True)
        #env = CentralFramestackObservationWrapper(env)
        env = CentralLocalPhaseSpecificObservationWrapper(env, unbalance=True)
        self.env = env

class MultiAttackCentralControlPhaseSpecificPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = BatterySingleRelativeInitPhaseSpecificDiscreteActionWrapper(env)
        env = CentralGlobalRewardWrapper(env, multi_attack=True)
        env = CentralLocalObservationWrapper(env, multi_attack=True)
        #env = CentralFramestackObservationWrapper(env)
        env = CentralLocalPhaseSpecificObservationWrapper(env, unbalance=True)
        self.env = env

class CentralControlPhaseSpecificContinuousPVInverterEnv(Wrapper):
    def __init__(self, **kwargs):
        env = CentralEnv(**kwargs)
        env = SingleRelativeInitPhaseSpecificContinuousActionWrapper(env)
        env = CentralGlobalRewardWrapper(env, multi_attack=True)
        env = CentralLocalObservationWrapper(env, multi_attack=True)
        env = CentralFramestackObservationWrapper(env)
        env = CentralLocalPhaseSpecificObservationWrapper(env, unbalance=True)
        self.env = env

import gym
from gym.spaces import Tuple, Discrete, Box
import pandas as pd
from pycigar.utils.opendss.pseudo_api import PyCIGAROpenDSSAPI
from collections import deque
import numpy as np
import math
from scipy import signal
import pycigar.utils.signal_processing as signal_processing
import random
from pycigar.utils.logging import logger

PF_CONVERTED = math.tan(math.acos(0.9))
STEP_BUFFER = 4
ACTION_RANGE = 0.1
ACTION_STEP = 0.01
DISCRETIZE_RELATIVE = int(ACTION_RANGE/ACTION_STEP)*2 + 1

class SimpleEnv(gym.Env):
    def __init__(self, sim_params, simulator='opendss'):

        # init
        self.LOAD_FACTOR = 2
        self.SOLAR_FACTOR = 2

        self.sim_params = sim_params
        df = pd.read_csv(sim_params['load_solar_path'], delimiter=',')
        load = df.iloc[:,:int(len(df.columns)/2)]
        self.load = load.reindex(sorted(load.columns), axis=1).to_numpy()
        solar = df.iloc[:,-int(len(df.columns)/2):]
        self.solar = solar.reindex(sorted(solar.columns), axis=1).to_numpy()

        df = pd.read_csv(sim_params['breakpoints_path'], delimiter=',')
        self.default_setting = df.reindex(sorted(df.columns), axis=1).to_numpy()
        self.device_list = [i.lower() for i in sorted(df.columns)] #sorted(df.columns)

        self.api = PyCIGAROpenDSSAPI()
        self.api.simulation_command('Redirect ' + self.sim_params['dss_path'])
        self.api.start_api()

        self.devices = Device(self)

        #work to do with action wrapper
        a_space = self.action_space
        self.a_size = sum(a.n for a in a_space)
        # relative action, init is centered
        self.init_action = [int(a.n / 2) for a in a_space]

        # change start time, percentage hack
        if self.sim_params['mode'] == 'training':
            self.mode = AttackChangeRandom()
        elif self.sim_params['mode'] == 'eval':
            self.mode = AttackChange()
        else:
            self.mode = False

    @property
    def action_space(self):
        return Tuple([Discrete(DISCRETIZE_RELATIVE)] * 3)

    @property
    def observation_space(self):
        return Box(low=-float('inf'), high=float('inf'), shape=(5 + DISCRETIZE_RELATIVE*3, ), dtype=np.float64)

    def reset(self):
        if not self.mode:
            self.hack_percentage = self.sim_params['hack_percentage']
            self.start_time = self.sim_params['start_time']
        else:
            self.hack_percentage, self.start_time = self.mode.change_mode()

        self.end_time = self.start_time + 750
        self.t = self.end_time - self.start_time

        self.time = 0

        self.v = deque([[0]*len(self.device_list)*2]*2, maxlen=2)

        # work with env reset
        self.api.simulation_command('Redirect ' + self.sim_params['dss_path'])
        self.api.start_api()

        solution_mode = 1#sim_params['simulation_config']['custom_configs'].get('solution_mode', 1)
        solution_number = 1#sim_params['simulation_config']['custom_configs'].get('solution_number', 1)
        solution_step_size = 1#sim_params['simulation_config']['custom_configs'].get('solution_step_size', 1)
        solution_control_mode = 2#sim_params['simulation_config']['custom_configs'].get('solution_control_mode', -1)
        solution_max_control_iterations = 1000000 #sim_params['simulation_config']['custom_configs'].get('solution_max_control_iterations', 1000000)
        solution_max_iterations = 30000#sim_params['simulation_config']['custom_configs'].get('solution_max_iterations', 30000)

        self.api.set_solution_mode(solution_mode)
        self.api.set_solution_number(solution_number)
        self.api.set_solution_step_size(solution_step_size)
        self.api.set_solution_control_mode(solution_control_mode)
        self.api.set_solution_max_control_iterations(solution_max_control_iterations)
        self.api.set_solution_max_iterations(solution_max_iterations)

        self.api.set_slack_bus_voltage(1.02)
        max_tap_change = 16
        forward_band = 1
        tap_delay = 2
        delay = 30
        tap_number = 0
        regs = self.api.get_all_regulator_names()
        for device_id in regs:
            # if device_id == 'feeder_rega':
            #     tap_number = 14
            # elif device_id == 'feeder_regb':
            #     tap_number = 13
            # elif device_id == 'feeder_regc':
            #     tap_number = 15
            # elif device_id == 'vreg2_a':
            #     tap_number = 7
            # elif device_id == 'vreg2_b':
            #     tap_number = 5
            # elif device_id == 'vreg2_c':
            #     tap_number = 4
            # elif device_id == 'vreg3_a':
            #     tap_number = 14
            # elif device_id == 'vreg3_b':
            #     tap_number = 10
            # elif device_id == 'vreg3_c':
            #     tap_number = 10
            # elif device_id == 'vreg4_a':
            #     tap_number = 9
            # elif device_id == 'vreg4_b':
            #     tap_number = 8
            # elif device_id == 'vreg4_c':
            #     tap_number = 11

            self.api.set_regulator_property(
                device_id,
                {
                    'max_tap_change': max_tap_change,
                    'forward_band': forward_band,
                    'tap_number': tap_number,
                    'tap_delay': tap_delay,
                    'delay': delay,
                },
            )

        if self.sim_params['log']:
            Logger = logger()
            Logger.reset()
            Logger.set_active(False)

        # call device to reset
        self.devices.reset()
        # run warm up
        for i, node in enumerate(self.device_list):
            node = node[:-3]
            self.api.set_node_kw(node, self.devices.p_actual[i])
            self.api.set_node_kvar(node, self.devices.q_actual[i])
        
        self.api.simulation_step()
        self.api.update_all_bus_voltages()
        vk = np.zeros(len(self.device_list))
        for i, device in enumerate(self.device_list):
            vk[i] = self.api.get_node_voltage(device[:-3])
        self.v.append(np.concatenate((vk,vk)))

        self.warm_up_k_step(50)
        
        if self.sim_params['log']:
            Logger.set_active(True)
            Logger.custom_metrics['start_time'] = self.start_time
            Logger.custom_metrics['hack_percentage'] = self.hack_percentage

        # OBSERVATION
        state = self.get_state()
        p_set = 0
        va = (state['v_worst'][0]-1)*10*2
        vb = (state['v_worst'][1]-1)*10*2
        vc = (state['v_worst'][2]-1)*10*2
        u_worst = state['u_worst'] * 10
        old_actions = self.init_action
        old_a_encoded = np.zeros(self.a_size)
        offsets = np.cumsum([0, *[a.n for a in self.action_space][:-1]])
        for action, offset in zip(old_actions, offsets):
            old_a_encoded[offset + action] = 1
        
        self.old_action = self.init_action
        
        return np.array([u_worst, p_set, *old_a_encoded, va, vb, vc])

    def step(self, action):
        done = False

        # AGENT ACTION CHANGE
        shift = np.array(action) * ACTION_STEP - ACTION_RANGE
        curve_a_translation = shift[0]
        curve_b_translation = shift[1]
        curve_c_translation = shift[2]

        def_shift = np.zeros_like(self.devices.def_mask)
        def_shift += (self.devices.def_mask == 0).astype(int) * curve_a_translation
        def_shift += (self.devices.def_mask == 1).astype(int) * curve_b_translation
        def_shift += (self.devices.def_mask == 2).astype(int) * curve_c_translation
        def_shift = np.expand_dims(def_shift, axis=1)
        
        hack_curve_all_translation = 0.0
        hack_curve_a_translation = -0.1
        hack_curve_b_translation = 0.1
        hack_curve_c_translation = -0.1

        observations = []
        for i in range(self.sim_params['sim_per_step']):
            if 250 < self.time < 500:
                adv_shift = np.zeros_like(self.devices.adv_mask)
                adv_shift += (self.devices.adv_mask == 0).astype(int) * hack_curve_a_translation
                adv_shift += (self.devices.adv_mask == 1).astype(int) * hack_curve_b_translation
                adv_shift += (self.devices.adv_mask == 2).astype(int) * hack_curve_c_translation
            else:
                adv_shift = np.zeros_like(self.devices.adv_mask)
            adv_shift = np.expand_dims(adv_shift, axis=1)

            self.devices.VBP = self.devices.VBP_init + def_shift + adv_shift

            self.devices.update()
            for i, node in enumerate(self.device_list):
                node = node[:-3]
                self.api.set_node_kw(node, self.devices.p_actual[i])
                self.api.set_node_kvar(node, self.devices.q_actual[i])
            self.api.simulation_step()
            self.api.update_all_bus_voltages()
            vk = np.zeros(len(self.device_list))
            for i, device in enumerate(self.device_list):
                vk[i] = self.api.get_node_voltage(device[:-3])
            self.v.append(np.concatenate((vk,vk)))

            self.time += 1
            observations.append(self.get_state())
            if self.time >= self.t:
                done = True
                break
        
        # OBSERVATION
        try:
            obs = {k: np.mean([d[k] for d in observations]) for k in observations[0]}
            obs['v_worst'] = observations[-1]['v_worst']
            obs['u_worst'] = observations[-1]['u_worst']
            obs['y_worst'] = observations[-1]['y_worst']
        except IndexError:
            obs = {'p_set_p_max': 0.0, 'sbar_solar_irr': 0.0, 'y': 0.0}
            obs['v_worst'] = [0, 0, 0]
            obs['u_worst'] = 0
            obs['u_mean'] = 0
            obs['y_worst'] = 0
            obs['y_mean'] = 0

        # OBSERVATION
        state = obs
        p_set = 1.5e-3 * state['sbar_solar_irr']
        va = (state['v_worst'][0]-1)*10*2
        vb = (state['v_worst'][1]-1)*10*2
        vc = (state['v_worst'][2]-1)*10*2
        u_worst = state['u_worst'] * 10 #state['u_mean'] * 50 #
        old_actions = self.old_action
        old_a_encoded = np.zeros(self.a_size)
        offsets = np.cumsum([0, *[a.n for a in self.action_space][:-1]])
        for a, offset in zip(old_actions, offsets):
            old_a_encoded[offset + a] = 1

        result_obs = np.array([u_worst, p_set, *old_a_encoded, va, vb, vc])

        # REWARD
        if action == self.old_action:
            roa = 0
        else:
            roa = 1
        r = 0
        r += - self.sim_params['M'] * state['u_worst']
        r += - self.sim_params['N'] * roa
        num_dev = int(self.devices.VBP.shape[0]/2)
        r += - self.sim_params['P'] * np.mean(np.linalg.norm(self.devices.VBP[:num_dev, :] - self.devices.VBP_init[:num_dev, :], axis = 1))
        r += - self.sim_params['Q'] * (1 - abs(state['p_set_p_max'])) ** 2

        self.old_action = action
        return result_obs, r, done, {}

    def warm_up_k_step(self, k):
        for _ in range(k):
            self.time += 1
            self.devices.update()
            for i, node in enumerate(self.device_list):
                node = node[:-3]
                self.api.set_node_kw(node, self.devices.p_actual[i])
                self.api.set_node_kvar(node, self.devices.q_actual[i])

            self.api.simulation_step()
            self.api.update_all_bus_voltages()
            vk = np.zeros(len(self.device_list))
            for i, device in enumerate(self.device_list):
                vk[i] = self.api.get_node_voltage(device[:-3])
            self.v.append(np.concatenate((vk,vk)))

    def get_state(self):
        u_worst, v_worst, u_mean, u_std, v_all, u_all_bus, load_to_bus = self.api.get_worst_u_node()

        y_worst = np.max(self.devices.y)
        y_mean = np.mean(self.devices.y)
        p_set_p_max = np.mean(self.devices.p_set / np.maximum(10, self.devices.solar_irr))
        sbar_solar_irr = np.mean((abs(self.devices.Sbar ** 2 - np.maximum(10, self.devices.solar_irr) ** 2)) ** (1 / 2))

        result = {'y_mean': y_mean,
                  'p_set_p_max': p_set_p_max,
                  'sbar_solar_irr': sbar_solar_irr}
        result['y_worst'] = y_worst
        result['v_worst'] = v_worst
        result['u_worst'] = u_worst
        result['u_mean'] = u_mean
        result['std'] = u_std

        # LOG
        if self.sim_params['log']:
            Logger = logger()
            #2129466b0a_pv
            Logger.log('a_metrics', 'a', self.devices.VBP[0, 0] - self.devices.VBP_init[0, 0])
            Logger.log('a_metrics', 'b', self.devices.VBP[1, 0] - self.devices.VBP_init[1, 0])
            Logger.log('a_metrics', 'c', self.devices.VBP[2, 0] - self.devices.VBP_init[2, 0])
            Logger.log('network', 'substation_power', self.api.get_total_power())
            Logger.log('u_metrics', 'u_worst', u_worst)
            Logger.log('u_metrics', 'u_mean', u_mean)
            for bus in u_all_bus:
                Logger.log('u_metrics', bus, u_all_bus[bus])
            for key in v_all:
                Logger.log('v_metrics', key, v_all[key])
            Logger.log('q_out', 'a', self.devices.q_out[:int(len(self.devices.q_out)/2)][self.devices.phases == 0])
            Logger.log('q_out', 'b', self.devices.q_out[:int(len(self.devices.q_out)/2)][self.devices.phases == 1])
            Logger.log('q_out', 'c', self.devices.q_out[:int(len(self.devices.q_out)/2)][self.devices.phases == 2])

            for reg in self.api.get_all_regulator_names():
                Logger.log('reg_metrics', reg, self.api.get_regulator_tap(reg))

        return result


class Device:
    def __init__(self, env):
        self.env = env
        ##################################################################
        VBP_init = self.env.default_setting.T   # (2354, 5)
        self.num_devices = num_devices = VBP_init.shape[0]   # 2354
        phases = self.phases = np.zeros(num_devices)
        for i, device in enumerate(self.env.device_list):
            if self.env.api.load_to_phase[device[:-3]] == 'a':
                phases[i] = 0
            elif self.env.api.load_to_phase[device[:-3]] == 'b':
                phases[i] = 1
            elif self.env.api.load_to_phase[device[:-3]] == 'c':
                phases[i] = 2
            else:
                phases[i] = -1

        self.def_mask = np.concatenate((phases, np.zeros(num_devices)-1))
        self.adv_mask = np.concatenate((np.zeros(num_devices)-1, phases))
        self.VBP = self.VBP_init = np.concatenate((VBP_init, VBP_init), axis=0)
        ####################################################################
        self.low_pass_filter_measure = 1.2
        self.low_pass_filter_output = 0.115
        self.lpf_high_pass_filter = 1
        self.lpf_low_pass_filter = 0.1
        self.lpf_delta_t = 1
        self.gain = 1e5
        self.solar_min_value = 0
        self.p_set = np.zeros(num_devices*2)
        self.q_set = np.zeros(num_devices*2)
        self.p_out = np.zeros(num_devices*2)
        self.q_out = np.zeros(num_devices*2)
        self.low_pass_filter_v = np.zeros(num_devices*2)
        self.lpf_psi = np.zeros(num_devices*2)
        self.lpf_epsilon = np.zeros(num_devices*2)
        self.lpf_y1 = np.zeros(num_devices*2)

        self.x = deque([[0]*num_devices*2]*15, maxlen=15)
        self.y = np.zeros(num_devices*2)

        self.p_out_new = np.zeros(num_devices*2)
        self.q_out_new = np.zeros(num_devices*2)

    def reset(self):
        num_devices = self.num_devices
        # update load, solar profiles
        self.load = np.concatenate((self.env.load[self.env.start_time:self.env.end_time]*(1-self.env.hack_percentage)*self.env.LOAD_FACTOR, self.env.load[self.env.start_time:self.env.end_time]*self.env.hack_percentage*self.env.LOAD_FACTOR), axis=1)
        self.solar = np.concatenate((self.env.solar[self.env.start_time:self.env.end_time]*(1-self.env.hack_percentage)*self.env.SOLAR_FACTOR, self.env.solar[self.env.start_time:self.env.end_time]*self.env.hack_percentage*self.env.SOLAR_FACTOR), axis=1)
        self.Sbar = np.concatenate((np.max(self.env.solar, axis=0)*(1-self.env.hack_percentage)*self.env.SOLAR_FACTOR, np.max(self.env.solar, axis=0)*self.env.hack_percentage*self.env.SOLAR_FACTOR))*1.1
        self.VBP = self.VBP_init
        self.p_actual = self.load[self.env.time]
        self.q_actual = self.load[self.env.time]*PF_CONVERTED
        self.p_actual = self.p_actual[:num_devices] + self.p_actual[-num_devices:]
        self.q_actual = self.q_actual[:num_devices] + self.q_actual[-num_devices:]

    def update(self):
        num_devices = self.num_devices
        pk = np.zeros(num_devices*2)
        qk = np.zeros(num_devices*2)
        q_avail = np.zeros(num_devices*2)
        self.solar_irr = self.solar[self.env.time]

        if not hasattr(self, 'step'):
            self.step = np.hstack((1 * np.ones(11), np.linspace(1, -1, 7), -1 * np.ones(11)))[:, None].T
            self.output_one = np.ones([num_devices*2,15])

        if self.env.time > 1:
            vk = self.env.v[1]
            vkm1 = self.env.v[0]

            vk = np.array(vk)
            vkm1 = np.array(vkm1)

            self.x.append(vk)
            output = np.array(self.x).T

            mask1 = np.max(output[:, STEP_BUFFER:-STEP_BUFFER], axis=1) - np.min(output[:, STEP_BUFFER:-STEP_BUFFER], axis=1) > 0.004
            norm_data = -1 + 2 * (output - np.min(output, axis=1)[:, None]) / (np.max(output, axis=1) - np.min(output, axis=1))[:, None]
            step_corr = signal.fftconvolve(norm_data, self.step, mode='valid', axes=1)
            mask2 = np.max(np.abs(step_corr), axis=1) > 10
            mask = mask1 & mask2

            filter_data = np.where(np.broadcast_to(mask[:, None], (mask.shape[0], 15)), self.output_one, output)[:, STEP_BUFFER:-STEP_BUFFER]


            lpf_psik = (filter_data[:, -1] - filter_data[:, -2] - (self.lpf_high_pass_filter * self.lpf_delta_t / 2 - 1) * self.lpf_psi) / \
                            (1 + self.lpf_high_pass_filter * self.lpf_delta_t / 2)
            self.lpf_psi = lpf_psik

            lpf_epsilonk = self.gain * (lpf_psik ** 2)

            y_value = (self.lpf_delta_t * self.lpf_low_pass_filter *
                    (lpf_epsilonk + self.lpf_epsilon) - (self.lpf_delta_t * self.lpf_low_pass_filter - 2) * self.lpf_y1) / \
                    (2 + self.lpf_delta_t * self.lpf_low_pass_filter)
            self.lpf_epsilon = lpf_epsilonk
            self.lpf_y1 = y_value
            self.y = y_value*0.04

            low_pass_filter_v = (self.lpf_delta_t * self.low_pass_filter_measure * (vk + vkm1) -
                                (self.lpf_delta_t * self.low_pass_filter_measure - 2) * (self.low_pass_filter_v)) / \
                                (2 + self.lpf_delta_t * self.low_pass_filter_measure)

            # compute p_set and q_set
            solar_idx = self.solar_irr >= self.solar_min_value

            idx = low_pass_filter_v <= self.VBP[:, 4]
            pk[idx] = -self.solar_irr[idx]
            q_avail[idx] = (self.Sbar[idx]**2 - pk[idx] ** 2) ** (1 / 2)

            idx = low_pass_filter_v <= self.VBP[:, 0]
            qk[idx] = -q_avail[idx]

            idx = (self.VBP[:, 0] < low_pass_filter_v) & (low_pass_filter_v <= self.VBP[:, 1])
            qk[idx] = q_avail[idx] / (self.VBP[:, 1][idx] - self.VBP[:, 0][idx]) * (low_pass_filter_v[idx] - self.VBP[:, 1][idx])

            idx = (self.VBP[:, 1] < low_pass_filter_v) & (low_pass_filter_v <= self.VBP[:, 2])
            qk[idx] = 0

            idx = (self.VBP[:, 2] < low_pass_filter_v) & (low_pass_filter_v <= self.VBP[:, 3])
            qk[idx] = q_avail[idx] / (self.VBP[:, 3][idx] - self.VBP[:, 2][idx]) * (low_pass_filter_v[idx] - self.VBP[:, 2][idx])

            idx = (self.VBP[:, 3] < low_pass_filter_v) & (low_pass_filter_v < self.VBP[:, 4])
            pk[idx] = -self.solar_irr[idx] / (self.VBP[:, 4][idx] - self.VBP[:, 3][idx]) * (self.VBP[:, 4][idx] - low_pass_filter_v[idx])
            qk[idx] = (self.Sbar[idx]**2 - pk[idx]**2)**(1/2)

            idx = low_pass_filter_v >= self.VBP[:, 4]
            pk[idx] = 0
            qk[idx] = self.Sbar[idx]

            pk[~solar_idx] = 0
            qk[~solar_idx] = 0


            # compute p_out and q_out
            self.p_out = ((self.lpf_delta_t * self.low_pass_filter_output * (pk + self.p_set) - (self.lpf_delta_t * self.low_pass_filter_output - 2) * (self.p_out)) / \
                            (2 + self.lpf_delta_t * self.low_pass_filter_output))
            self.q_out = ((self.lpf_delta_t * self.low_pass_filter_output * (qk + self.q_set) - (self.lpf_delta_t * self.low_pass_filter_output - 2) * (self.q_out)) / \
                            (2 + self.lpf_delta_t * self.low_pass_filter_output))

            #self.p_out_new = self.p_out_new + self.lpf_delta_t*np.sign(pk-self.p_out_new)*np.minimum((self.p_ramp_rate*self.Sbar), np.abs(pk-self.p_out_new))
            #self.q_out_new = self.q_out_new + self.lpf_delta_t*np.sign(qk-self.q_out_new)*np.minimum((self.q_ramp_rate*self.Sbar), np.abs(pk-self.q_out_new))
            #self.p_out = self.p_out + self.lpf_delta_t*np.sign(pk-self.p_out)*np.minimum((self.p_ramp_rate*self.Sbar), np.abs(pk-self.p_out))
            #self.q_out = self.q_out + self.lpf_delta_t*np.sign(qk-self.q_out)*np.minimum((self.q_ramp_rate*self.Sbar), np.abs(pk-self.q_out))

            self.p_set = pk
            self.q_set = qk
            self.low_pass_filter_v = low_pass_filter_v
        # import old V to x
        # inject to node


        self.p_actual = self.load[self.env.time] + self.p_out
        self.q_actual = self.load[self.env.time]*PF_CONVERTED + self.q_out

        self.p_actual = self.p_actual[:num_devices] + self.p_actual[-num_devices:]
        self.q_actual = self.q_actual[:num_devices] + self.q_actual[-num_devices:]

class AttackChangeRandom:
    def __init__(self):
        self.mode = 0
        percentage = np.linspace(10, 50, 9) / 100
        start_time = np.linspace(100, 11000, 10).astype(int)
        self.scenarios = []
        for p in percentage:
            for s in start_time:
                s = int(s)
                scenarios = [p, s]
                self.scenarios.append(scenarios)
    def change_mode(self):
        res = self.scenarios[self.mode]
        self.mode = random.randint(0, len(self.scenarios) - 1)
        return res

class AttackChange:
    def __init__(self):
        self.mode = 0
        percentage = np.linspace(10, 40, 4) / 100
        start_time = np.linspace(100, 11000, 2).astype(int)
        self.scenarios = []
        for p in percentage:
            for s in start_time:
                s = int(s)
                scenarios = [p, s]
                self.scenarios.append(scenarios)
    def change_mode(self):
        res = self.scenarios[self.mode]
        self.mode += 1
        if self.mode == len(self.scenarios):
            self.mode = 0
        return res
