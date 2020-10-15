from collections import deque

import numpy as np
from scipy import signal

import pycigar.utils.signal_processing as signal_processing
from pycigar.devices.base_device import BaseDevice

from pycigar.utils.logging import logger
from copy import copy

DEFAULT_CONTROL_SETTING = [0.98, 1.01, 1.02, 1.05, 1.07]
STEP_BUFFER = 4


class VectorizedPVDevice:
    def __init__(self, k):
        """Instantiate an PV device."""
        self.k = k
        self.is_disable_log = k.sim_params['is_disable_log']
        self.list_device = k.device.get_pv_device_ids()
        self.list_node = [k.device.get_node_connected_to(device_id) for device_id in self.list_device]

        self.Sbar = []
        self.solar_generation = []
        self.VBP = []
        self.low_pass_filter_measure = []
        self.low_pass_filter_output = []
        self.p_ramp_rate = []
        self.q_ramp_rate = []
        for device_id in self.list_device:
            device = k.device.get_device(device_id)
            self.solar_generation.append(device.solar_generation)
            self.VBP.append(device.control_setting)
            self.low_pass_filter_measure.append(device.low_pass_filter_measure)
            self.low_pass_filter_output.append(device.low_pass_filter_output)
            self.Sbar.append(device.Sbar)
            self.p_ramp_rate.append(device.p_ramp_rate)
            self.q_ramp_rate.append(device.q_ramp_rate)

        self.solar_generation = np.array(self.solar_generation)
        self.VBP = np.array(self.VBP)
        self.low_pass_filter_measure = np.array(self.low_pass_filter_measure)
        self.low_pass_filter_output = np.array(self.low_pass_filter_output)
        self.Sbar = np.array(self.Sbar)
        self.p_ramp_rate = np.array(self.p_ramp_rate)
        self.q_ramp_rate = np.array(self.q_ramp_rate)

        self.lpf_high_pass_filter = 1
        self.lpf_low_pass_filter = 0.1
        self.lpf_delta_t = 1
        self.gain = 1e5
        self.solar_min_value = 0

        self.p_set = np.array([0]*len(self.list_device))
        self.q_set = np.array([0]*len(self.list_device))
        self.p_out = np.array([0]*len(self.list_device))
        self.q_out = np.array([0]*len(self.list_device))
        self.low_pass_filter_v = np.array([0]*len(self.list_device))
        self.lpf_psi = np.array([0]*len(self.list_device))
        self.lpf_epsilon = np.array([0]*len(self.list_device))
        self.lpf_y1 = np.array([0]*len(self.list_device))

        self.x = deque([[0]*len(self.list_device)]*15, maxlen=15)

        self.solar_irr = None
        self.y = np.array([0]*len(self.list_device))

        self.p_out_new = np.array([0]*len(self.list_device))
        self.q_out_new = np.array([0]*len(self.list_device))


    def update(self, k):

        self.solar_irr = self.solar_generation[:, k.time]
        pk = np.zeros(len(self.list_device))
        qk = np.zeros(len(self.list_device))
        q_avail = np.zeros(len(self.list_device))

        if not hasattr(self, 'step'):
            self.step = np.hstack((1 * np.ones(11), np.linspace(1, -1, 7), -1 * np.ones(11)))[:, None].T
            self.output_one = np.ones([len(self.list_device),15])

        if k.time > 1:
            vk = []
            vkm1 = []
            for node_id in self.list_node:
                vk.append(abs(k.node.nodes[node_id]['voltage'][k.time - 1]))
                vkm1.append(abs(k.node.nodes[node_id]['voltage'][k.time - 2]))

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
            if 's701a' in k.node.nodes and 's701b' in k.node.nodes and 's701c' in k.node.nodes:
                va = abs(k.node.nodes['s701a']['voltage'][k.time - 1])
                vb = abs(k.node.nodes['s701b']['voltage'][k.time - 1])
                vc = abs(k.node.nodes['s701c']['voltage'][k.time - 1])
                mean = (va+vb+vc)/3
                max_diff = max(abs(va - mean), abs(vb - mean), abs(vc - mean))
                self.u = max_diff / mean

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

        for i, device_id in enumerate(self.list_device):
            k.node.nodes[self.list_node[i]]['PQ_injection']['P'] += self.p_out[i]
            k.node.nodes[self.list_node[i]]['PQ_injection']['Q'] += self.q_out[i]

        self.log()

    def reset(self):
        """See parent class."""
        self.__init__(self.k)

    def set_control_setting(self, device_id, control_setting):
        """See parent class."""
        idx = self.list_device.index(device_id)
        self.VBP[idx] = control_setting

    def log(self):
        # log history
        if not self.is_disable_log:
            Logger = logger()
            for i, device_id in enumerate(self.list_device):
                Logger.log(device_id, 'y', self.y[i])

                #Logger.log(self.device_id, 'u', self.u[i])
                Logger.log(device_id, 'p_set', self.p_set[i])
                Logger.log(device_id, 'q_set', self.q_set[i])
                Logger.log(device_id, 'p_out', self.p_out[i])
                Logger.log(device_id, 'q_out', self.q_out[i])
                Logger.log(device_id, 'p_out_new', self.p_out_new[i])
                Logger.log(device_id, 'q_out_new', self.q_out_new[i])
                Logger.log(device_id, 'control_setting', copy(self.VBP[i]))
                if self.Sbar != []:
                    Logger.log(device_id, 'sbar_solarirr', 1.5e-3*(abs(self.Sbar[i] ** 2 - max(10, self.solar_irr[i]) ** 2)) ** (1 / 2))
                    Logger.log(device_id, 'sbar_pset', self.p_set[i] / self.Sbar[i])

                Logger.log(device_id, 'solar_irr', self.solar_irr[i])