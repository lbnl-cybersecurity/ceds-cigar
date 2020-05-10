from collections import deque

import numpy as np

import pycigar.utils.signal_processing as signal_processing
from pycigar.devices.base_device import BaseDevice

from pycigar.utils.logging import logger

DEFAULT_CONTROL_SETTING = np.array([0.98, 1.01, 1.02, 1.05, 1.07])
step_buffer = 4


class PVDevice(BaseDevice):
    def __init__(self, device_id, additional_params):
        """Instantiate an PV device."""
        BaseDevice.__init__(
            self,
            device_id,
            additional_params
        )
        self.solar_generation = None

        self.control_setting = additional_params.get('default_control_setting', DEFAULT_CONTROL_SETTING)
        self.low_pass_filter_measure = additional_params.get('low_pass_filter_measure', 1)
        self.low_pass_filter_output = additional_params.get('low_pass_filter_output', 0.1)
        self.low_pass_filter = additional_params.get('low_pass_filter', 0.1)
        self.high_pass_filter = additional_params.get('high_pass_filter', 1)
        self.gain = additional_params.get('gain', 1e5)
        self.delta_t = additional_params.get('delta_t', 1)
        self.solar_min_value = additional_params.get('solar_min_value', 5)
        Logger = logger()
        if 'init_control_settings' in Logger.custom_metrics:
            logger().custom_metrics['init_control_settings'][device_id] = np.array(self.control_setting)
        else:
            logger().custom_metrics['init_control_settings'] = {device_id: np.array(self.control_setting)}

        self.p_set = np.zeros(2)
        self.q_set = np.zeros(2)
        self.p_out = np.zeros(2)
        self.q_out = np.zeros(2)
        self.low_pass_filter_v = np.zeros(2)

        self.solar_irr = 0
        self.psi = 0
        self.epsilon = 0
        self.y = 0
        self.u = 0

        # init for signal processing on Voltage
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
        self.x = deque([0] * (len(self.BP1z[0, :]) + step_buffer * 2), maxlen=(len(self.BP1z[0, :]) + step_buffer * 2))

    def update(self, k):
        """See parent class."""
        # TODO: eliminate this magic number
        self.Sbar = 1.1 * np.max(self.solar_generation)
        VBP = self.control_setting

        # record voltage magnitude measurement
        node_id = k.device.get_node_connected_to(self.device_id)
        self.node_id = node_id
        Sbar = self.Sbar
        solar_irr = self.solar_generation[k.time]
        self.solar_irr = solar_irr
        solar_minval = self.solar_min_value
        step = np.hstack((1 * np.ones(11), np.linspace(1, -1, 7), -1 * np.ones(11)))
        if k.time > 1:
            vk = np.abs(k.node.nodes[node_id]['voltage'][k.time - 1])
            vkm1 = np.abs(k.node.nodes[node_id]['voltage'][k.time - 2])
            self.v_meas_k = vk
            self.v_meas_km1 = vkm1
            self.x.append(vk)
            # print(" \n X \n")
            # print(self.x)
            output = np.array(self.x).copy()
            # if k.time > 12:
            if np.max(output[step_buffer:-step_buffer]) - np.min(output[step_buffer:-step_buffer]) > 0.004:
                norm_data = -1 + 2 * (list(output) - np.min(list(output))) / (np.max(list(output)) - np.min(list(output)))
                step_corr = np.convolve(norm_data, step, mode='valid')
                if np.max(abs(step_corr)) > 10:
                    output = np.ones(15)
            filter_data = output[step_buffer:-step_buffer]

            self.y1.append(1 / self.BP1z[1, -1] * (np.sum(-self.BP1z[1, 0:-1] * self.y1) + np.sum(self.BP1z[0, :] * filter_data)))
            self.y2.append(self.y1[-1]**2)
            self.y3.append(1 / self.LPF2z[1, -1] * (np.sum(-self.LPF2z[1, 0:-1] * self.y3) + np.sum(self.LPF2z[0, :] * self.y2)))

            self.y = max(1e4 * self.y3[-1], 0)

            if 's701a' in k.node.nodes and 's701b' in k.node.nodes and 's701c' in k.node.nodes:
                va = np.abs(k.node.nodes['s701a']['voltage'][k.time - 1])
                vb = np.abs(k.node.nodes['s701b']['voltage'][k.time - 1])
                vc = np.abs(k.node.nodes['s701c']['voltage'][k.time - 1])
                mean = (va+vb+vc)/3
                max_diff = max([abs(va - mean), abs(vb - mean), abs(vc - mean)])
                self.u = max_diff / mean

        T = self.delta_t
        lpf_m = self.low_pass_filter_measure
        lpf_o = self.low_pass_filter_output

        pk = 0
        qk = 0
        if k.time > 1:
            # compute v_lpf (lowpass filter)
            # gammakcalc = (T*lpf*(Vmagk + Vmagkm1) - (T*lpf - 2)*gammakm1)
            # /(2 + T*lpf)
            low_pass_filter_v = self.low_pass_filter_v[1] = (T * lpf_m * (self.v_meas_k + self.v_meas_km1) -
                                                             (T * lpf_m - 2) * (self.low_pass_filter_v[0])) / \
                                                            (2 + T * lpf_m)

            # compute p_set and q_set
            if solar_irr >= solar_minval:
                if low_pass_filter_v <= VBP[4]:
                    # no curtailment
                    pk = -solar_irr
                    q_avail = (Sbar ** 2 - pk ** 2) ** (1 / 2)

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
                        d = -solar_irr / (VBP[4] - VBP[3])
                        pk = d * (VBP[4] - low_pass_filter_v)
                        qk = (Sbar ** 2 - pk ** 2) ** (1 / 2)
                elif low_pass_filter_v >= VBP[4]:
                    # full real power curtailment for VAR support
                    pk = 0
                    qk = Sbar

            self.p_set[1] = pk
            self.q_set[1] = qk

            # compute p_out and q_out
            self.p_out[1] = (T * lpf_o * (self.p_set[1] + self.p_set[0]) - (T * lpf_o - 2) * (self.p_out[0])) / \
                            (2 + T * lpf_o)
            self.q_out[1] = (T * lpf_o * (self.q_set[1] + self.q_set[0]) - (T * lpf_o - 2) * (self.q_out[0])) / \
                            (2 + T * lpf_o)

            self.p_set[0] = self.p_set[1]
            self.q_set[0] = self.q_set[1]
            self.p_out[0] = self.p_out[1]
            self.q_out[0] = self.q_out[1]
            self.low_pass_filter_v[0] = self.low_pass_filter_v[1]

        # import old V to x
        # inject to node
        k.node.nodes[node_id]['PQ_injection']['P'] += self.p_out[1]
        k.node.nodes[node_id]['PQ_injection']['Q'] += self.q_out[1]

        # log necessary info
        self.log()

    def reset(self):
        """See parent class."""
        self.__init__(self.device_id, self.init_params)
        self.log()

    def set_control_setting(self, control_setting):
        """See parent class."""
        self.control_setting = control_setting

    def log(self):
        # log history
        Logger = logger()
        Logger.log(self.device_id, 'y', self.y)
        Logger.log(self.device_id, 'u', self.u)
        Logger.log(self.device_id, 'p_set', self.p_set[1])
        Logger.log(self.device_id, 'q_set', self.q_set[1])
        Logger.log(self.device_id, 'p_out', self.p_out[1])
        Logger.log(self.device_id, 'q_out', self.q_out[1])
        Logger.log(self.device_id, 'control_setting', self.control_setting)
        if hasattr(self, 'Sbar'):
            Logger.log(self.device_id, 'sbar_solarirr', 1.5e-3*(abs(self.Sbar ** 2 - max(10, self.solar_irr) ** 2)) ** (1 / 2))
            Logger.log(self.device_id, 'sbar_pset', self.p_set[1] / self.Sbar)

        Logger.log(self.device_id, 'solar_irr', self.solar_irr)
        if hasattr(self, 'node_id'):
            Logger.log_single(self.device_id, 'node', self.node_id)
