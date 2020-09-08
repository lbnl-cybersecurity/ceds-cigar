from collections import deque

import numpy as np

import pycigar.utils.signal_processing as signal_processing
from pycigar.devices.base_device import BaseDevice

from pycigar.utils.logging import logger

DEFAULT_CONTROL_SETTING = [0.98, 1.01, 1.02, 1.05, 1.07]
STEP_BUFFER = 4


class PVDevice(BaseDevice):
    def __init__(self, device_id, additional_params, is_disable_log=False):
        """Instantiate an PV device."""
        BaseDevice.__init__(self, device_id, additional_params)
        self.is_disable_log = is_disable_log
        self.solar_generation = None
        self.Sbar = None

        self.control_setting = additional_params.get('default_control_setting', DEFAULT_CONTROL_SETTING)
        self.low_pass_filter_measure_mean = additional_params.get('low_pass_filter_measure_mean', 1.2)
        self.low_pass_filter_output_mean = additional_params.get('low_pass_filter_output_mean', 0.1)
        self.low_pass_filter_measure_std = additional_params.get('low_pass_filter_measure_std', 0)
        self.low_pass_filter_output_std = additional_params.get('low_pass_filter_output_std', 0)

        self.low_pass_filter_measure = self.low_pass_filter_measure_std * np.random.randn() + self.low_pass_filter_measure_mean
        self.low_pass_filter_output = self.low_pass_filter_output_std * np.random.randn() + self.low_pass_filter_output_mean

        self.gain = additional_params.get('gain', 1e5)
        self.delta_t = additional_params.get('delta_t', 1)
        self.solar_min_value = additional_params.get('solar_min_value', 5)
        self.is_butterworth_filter = additional_params.get('is_butterworth_filter', True)
        self.p_ramp_rate = additional_params.get('p_ramp_rate', 1)
        self.q_ramp_rate = additional_params.get('q_ramp_rate', 0.25)

        Logger = logger()
        if 'init_control_settings' in Logger.custom_metrics:
            logger().custom_metrics['init_control_settings'][device_id] = self.control_setting
        else:
            logger().custom_metrics['init_control_settings'] = {device_id: self.control_setting}

        self.p_set = deque([0] * 2, maxlen=2)
        self.q_set = deque([0] * 2, maxlen=2)
        self.p_out = deque([0] * 2, maxlen=2)
        self.q_out = deque([0] * 2, maxlen=2)
        self.low_pass_filter_v = deque([0] * 2, maxlen=2)

        self.solar_irr = 0
        self.psi = 0
        self.epsilon = 0
        self.y = 0
        self.u = 0

        self.custom_control_setting = {}

        # init for signal processing on Voltage
        if self.is_butterworth_filter:
            Ts = 1
            fosc = 0.15
            hp1, temp = signal_processing.butterworth_highpass(2, 2 * np.pi * fosc / 1)
            lp1, temp = signal_processing.butterworth_lowpass(4, 2 * np.pi * 1 * fosc)
            bp1num = np.convolve(hp1[0, :], lp1[0, :])
            bp1den = np.convolve(hp1[1, :], lp1[1, :])
            bp1s = np.array([bp1num, bp1den])
            self.BP1z = signal_processing.c2dbilinear(bp1s, Ts)
            lpf2, temp = signal_processing.butterworth_lowpass(2, 2 * np.pi * fosc / 2)
            self.LPF2z = signal_processing.c2dbilinear(lpf2, Ts)
            self.nbp1 = self.BP1z.shape[1] - 1
            self.nlpf2 = self.LPF2z.shape[1] - 1

            self.y1 = deque([0] * len(self.BP1z[1, 0:-1]), maxlen=len(self.BP1z[1, 0:-1]))
            self.y2 = deque([0] * len(self.LPF2z[0, :]), maxlen=len(self.LPF2z[0, :]))
            self.y3 = deque([0] * len(self.LPF2z[1, 0:-1]), maxlen=len(self.LPF2z[1, 0:-1]))
            # self.x = deque([0] * (len(self.BP1z[0, :]) + STEP_BUFFER * 2), maxlen=(len(self.BP1z[0, :]) + STEP_BUFFER * 2))
            self.x = deque([0] * 15, maxlen=15)

        else:
            self.lpf_psi = deque([0] * 2, maxlen=2)
            self.lpf_epsilon = deque([0] * 2, maxlen=2)
            self.lpf_y1 = deque([0] * 2, maxlen=2)
            self.lpf_high_pass_filter = 1
            self.lpf_low_pass_filter = 0.1
            self.lpf_x = deque([0] * 15, maxlen=15)
            self.lpf_delta_t = self.delta_t
            self.lpf_y = 0
            self.x = deque([0] * 15, maxlen=15)

    def update(self, k):
        """See parent class."""
        VBP = self.control_setting
        # record voltage magnitude measurement
        if not hasattr(self, 'node_id'):
            self.node_id = k.device.get_node_connected_to(self.device_id)

        self.solar_irr = self.solar_generation[k.time]

        if not hasattr(self, 'step'):
            self.step = np.hstack((1 * np.ones(11), np.linspace(1, -1, 7), -1 * np.ones(11)))

        if k.time > 1:
            vk = abs(k.node.nodes[self.node_id]['voltage'][k.time - 1])
            vkm1 = abs(k.node.nodes[self.node_id]['voltage'][k.time - 2])

            if k.sim_params['is_disable_y'] == True:
                y = 0
            else:
                # self.x.append(vk)
                if k.time >= 16:
                    output = abs(k.node.nodes[self.node_id]['voltage'][k.time - 16 : k.time - 1])
                else:
                    self.x.append(vk)
                    output = np.array(self.x)

                if np.max(output[STEP_BUFFER:-STEP_BUFFER]) - np.min(output[STEP_BUFFER:-STEP_BUFFER]) > 0.004:
                    norm_data = -1 + 2 * (output - np.min(output)) / (np.max(output) - np.min(output))
                    step_corr = np.convolve(norm_data, self.step, mode='valid')

                    if max(abs(step_corr)) > 10:
                        output = np.ones(15)
                filter_data = output[STEP_BUFFER:-STEP_BUFFER]

                if self.is_butterworth_filter:
                    self.y1.append(
                        1
                        / self.BP1z[1, -1]
                        * (np.sum(-self.BP1z[1, 0:-1] * self.y1) + np.sum(self.BP1z[0, :] * filter_data))
                    )
                    self.y2.append(self.y1[-1] ** 2)
                    self.y3.append(
                        1
                        / self.LPF2z[1, -1]
                        * (np.sum(-self.LPF2z[1, 0:-1] * self.y3) + np.sum(self.LPF2z[0, :] * self.y2))
                    )

                    self.y = max(1e4 * self.y3[-1], 0)
                else:
                    lpf_psik = (
                        filter_data[-1]
                        - filter_data[-2]
                        - (self.lpf_high_pass_filter * self.lpf_delta_t / 2 - 1) * self.lpf_psi[1]
                    ) / (1 + self.lpf_high_pass_filter * self.lpf_delta_t / 2)
                    self.lpf_psi.append(lpf_psik)

                    lpf_epsilonk = self.gain * (lpf_psik ** 2)
                    self.lpf_epsilon.append(lpf_epsilonk)

                    y_value = (
                        self.lpf_delta_t * self.lpf_low_pass_filter * (self.lpf_epsilon[1] + self.lpf_epsilon[0])
                        - (self.lpf_delta_t * self.lpf_low_pass_filter - 2) * self.lpf_y1[1]
                    ) / (2 + self.lpf_delta_t * self.lpf_low_pass_filter)
                    self.lpf_y1.append(y_value)
                    self.y = y_value * 0.04 * 3

            if 's701a' in k.node.nodes and 's701b' in k.node.nodes and 's701c' in k.node.nodes:
                va = abs(k.node.nodes['s701a']['voltage'][k.time - 1])
                vb = abs(k.node.nodes['s701b']['voltage'][k.time - 1])
                vc = abs(k.node.nodes['s701c']['voltage'][k.time - 1])
                mean = (va + vb + vc) / 3
                max_diff = max(abs(va - mean), abs(vb - mean), abs(vc - mean))
                self.u = max_diff / mean

                v14a = abs(k.node.nodes['s714a']['voltage'][k.time - 1])
                v14b = abs(k.node.nodes['s714b']['voltage'][k.time - 1])
                mean = (v14a + v14b) / 2
                max_diff = max(abs(v14a - mean), abs(v14b - mean))
                self.u += max_diff / mean

                v22b = abs(k.node.nodes['s722b']['voltage'][k.time - 1])
                v22c = abs(k.node.nodes['s722c']['voltage'][k.time - 1])
                mean = (v22b + v22c) / 2
                max_diff = max(abs(v22c - mean), abs(v22b - mean))
                self.u += max_diff / mean
                
                v42a = abs(k.node.nodes['s742a']['voltage'][k.time - 1])
                v42b = abs(k.node.nodes['s742b']['voltage'][k.time - 1])
                mean = (v42a + v42b) / 2
                max_diff = max(abs(v42a - mean), abs(v42b - mean))
                self.u += max_diff / mean
                
                self.u /= 4.0

        T = self.delta_t
        lpf_m = self.low_pass_filter_measure
        lpf_o = self.low_pass_filter_output

        pk = 0
        qk = 0
        if k.time > 1:
            low_pass_filter_v = (T * lpf_m * (vk + vkm1) - (T * lpf_m - 2) * (self.low_pass_filter_v[1])) / (
                2 + T * lpf_m
            )

            if 'v_offset' in self.custom_control_setting:
                low_pass_filter_v += self.custom_control_setting['v_offset']

            # compute p_set and q_set
            if self.solar_irr >= self.solar_min_value:
                if low_pass_filter_v <= VBP[4]:
                    # no curtailment
                    pk = -self.solar_irr
                    q_avail = (self.Sbar ** 2 - pk ** 2) ** (1 / 2)

                    # determine VAR support
                    if low_pass_filter_v <= VBP[0]:
                        # inject all available var
                        qk = -q_avail
                    elif VBP[0] < low_pass_filter_v <= VBP[1]:
                        # partial VAR injection
                        c = q_avail / (VBP[1] - VBP[0])
                        qk = c * (low_pass_filter_v - VBP[1])
                    elif VBP[1] < low_pass_filter_v <= VBP[2]:
                        # No var support
                        qk = 0
                    elif VBP[2] < low_pass_filter_v < VBP[3]:
                        # partial Var consumption
                        c = q_avail / (VBP[3] - VBP[2])
                        qk = c * (low_pass_filter_v - VBP[2])
                    elif VBP[3] < low_pass_filter_v < VBP[4]:
                        # partial real power curtailment
                        d = -self.solar_irr / (VBP[4] - VBP[3])
                        pk = d * (VBP[4] - low_pass_filter_v)
                        qk = (self.Sbar ** 2 - pk ** 2) ** (1 / 2)
                elif low_pass_filter_v >= VBP[4]:
                    # full real power curtailment for VAR support
                    pk = 0
                    qk = self.Sbar

            self.p_set.append(pk)
            self.q_set.append(qk)

            # compute p_out and q_out
            self.p_out.append(
                (T * lpf_o * (self.p_set[1] + self.p_set[0]) - (T * lpf_o - 2) * (self.p_out[1])) / (2 + T * lpf_o)
            )
            self.q_out.append(
                (T * lpf_o * (self.q_set[1] + self.q_set[0]) - (T * lpf_o - 2) * (self.q_out[1])) / (2 + T * lpf_o)
            )

            self.low_pass_filter_v.append(low_pass_filter_v)

        # import old V to x
        # inject to node
        k.node.nodes[self.node_id]['PQ_injection']['P'] += self.p_out[1]
        k.node.nodes[self.node_id]['PQ_injection']['Q'] += self.q_out[1]

        # log necessary info
        self.log()

    def reset(self):
        """See parent class."""
        self.__init__(self.device_id, self.init_params)
        self.log()

    def set_control_setting(self, control_setting, custom_control_setting=None):
        """See parent class."""
        if control_setting is not None:
            self.control_setting = control_setting
        if custom_control_setting:
            self.custom_control_setting = custom_control_setting

    def log(self):
        # log history
            if not self.is_disable_log:
                Logger = logger()
                Logger.log(self.device_id, 'y', self.y)
                Logger.log(self.device_id, 'u', self.u)
                Logger.log(self.device_id, 'p_set', self.p_set[1])
                Logger.log(self.device_id, 'q_set', self.q_set[1])
                Logger.log(self.device_id, 'p_out', self.p_out[1])
                Logger.log(self.device_id, 'q_out', self.q_out[1])
                Logger.log(self.device_id, 'control_setting', self.control_setting)
                if self.Sbar:
                    Logger.log(self.device_id, 'sbar_solarirr', 1.5e-3*(abs(self.Sbar ** 2 - max(10, self.solar_irr) ** 2)) ** (1 / 2))
                    Logger.log(self.device_id, 'sbar_pset', self.p_set[1] / self.Sbar)
                    Logger.log(self.device_id, 'pset_pmax', self.p_set[1] / max(10, self.solar_irr))
                    Logger.log(self.device_id, 'q_avail_real', abs(self.Sbar ** 2 - self.p_out[1] ** 2) ** (1 / 2)*np.sign(self.q_out[1]))
                Logger.log(self.device_id, 'solar_irr', self.solar_irr)
                if hasattr(self, 'node_id'):
                    Logger.log_single(self.device_id, 'node', self.node_id)
