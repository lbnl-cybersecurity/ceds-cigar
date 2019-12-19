import numpy as np
from pycigar.devices.base_device import BaseDevice

DEFAULT_CONTROL_SETTING = np.array([0.98, 1.01, 1.02, 1.05, 1.07])


class RegulatorDevice(BaseDevice):
    def __init__(self, device_id, additional_params=None):
        """Instantiate an PV device."""
        BaseDevice.__init__(self, device_id, additional_params)


        if "default_control_setting" in additional_params:
            self.control_setting = additional_params["default_control_setting"]
        else:
            self.control_setting = DEFAULT_CONTROL_SETTING

        if "low_pass_filter_measure" in additional_params:
            self.low_pass_filter_measure = additional_params["low_pass_filter_measure"]
        else:
            self.low_pass_filter_measure = 1
        if "low_pass_filter_output" in additional_params:
            self.low_pass_filter_output = additional_params["low_pass_filter_output"]
        else:
            self.low_pass_filter_output = 0.1

        if 'low_pass_filter' in additional_params:
            self.low_pass_filter = additional_params['low_pass_filter']
        else:
            self.low_pass_filter = 0.1

        if 'high_pass_filter' in additional_params:
            self.high_pass_filter = additional_params['high_pass_filter']
        else:
            self.high_pass_filter = 1

        if 'gain' in additional_params:
            self.gain = additional_params['gain']
        else:
            self.gain = 1e5

        if "delta_t" in additional_params:
            self.delta_t = additional_params["delta_t"]
        else:
            self.delta_t = 1

        if "solar_min_value" in additional_params:
            self.solar_min_value = additional_params["solar_min_value"]
        else:
            self.solar_min_value = 5

        self.p_set = np.zeros(2)
        self.q_set = np.zeros(2)
        self.p_out = np.zeros(2)
        self.q_out = np.zeros(2)
        self.low_pass_filter_v = np.zeros(2)

        self.psi = 0
        self.epsilon = 0
        self.y = 0

    def update(self, k):
        """See parent class."""
        # TODO: eliminate this magic number
        self.Sbar = 1.1 * np.max(self.solar_generation)
        VBP = self.control_setting

        # record voltage magnitude measurement
        node_id = k.device.get_node_connected_to(self.device_id)
        Sbar = self.Sbar
        solar_irr = self.solar_generation[k.time]
        solar_minval = self.solar_min_value
        if k.time > 1:
            vk = np.abs(k.node.nodes[node_id]['voltage'][k.time-1])
            vkm1 = np.abs(k.node.nodes[node_id]['voltage'][k.time-2])
            self.v_meas_k = vk
            self.v_meas_km1 = vkm1
            psikm1 = self.psi
            epsilonkm1 = self.epsilon
            ykm1 = self.y
            self.psi = psik = (vk - vkm1 - (self.high_pass_filter * self.delta_t / 2 - 1) * psikm1) / \
                              (1 + self.high_pass_filter * self.delta_t / 2)
            self.epsilon = epsilonk = self.gain * (psik ** 2)
            self.y = (self.delta_t * self.low_pass_filter *
                      (epsilonk + epsilonkm1) - (self.delta_t * self.low_pass_filter - 2) * ykm1) / \
                     (2 + self.delta_t * self.low_pass_filter)

        T = self.delta_t
        lpf_m = self.low_pass_filter_measure
        lpf_o = self.low_pass_filter_output

        pk = 0
        qk = 0
        if (k.time > 1):

            # compute v_lpf (lowpass filter)
            # gammakcalc = (T*lpf*(Vmagk + Vmagkm1) - (T*lpf - 2)*gammakm1)
            # /(2 + T*lpf)
            low_pass_filter_v = self.low_pass_filter_v[1] = (T * lpf_m * (self.v_meas_k + self.v_meas_km1) -
                                                             (T * lpf_m - 2) * (self.low_pass_filter_v[0])) / \
                                                            (2 + T * lpf_m)

            # compute p_set and q_set
            if (solar_irr >= solar_minval):
                if (low_pass_filter_v <= VBP[4]):
                    # no curtailment
                    pk = -solar_irr
                    q_avail = (Sbar ** 2 - pk ** 2) ** (1 / 2)

                    # determine VAR support
                    if (low_pass_filter_v <= VBP[0]):
                        # inject all available var
                        qk = -q_avail
                    elif (low_pass_filter_v > VBP[0]
                          and low_pass_filter_v <= VBP[1]):
                        # partial VAR injection
                        c = q_avail / (VBP[1] - VBP[0])
                        qk = c * (low_pass_filter_v - VBP[1])
                    elif (low_pass_filter_v > VBP[1]
                          and low_pass_filter_v <= VBP[2]):
                        # No var support
                        qk = 0
                    elif (low_pass_filter_v > VBP[2]
                          and low_pass_filter_v < VBP[3]):
                        # partial Var consumption
                        c = q_avail / (VBP[3] - VBP[2])
                        qk = c * (low_pass_filter_v - VBP[2])
                    elif (low_pass_filter_v > VBP[3]
                          and low_pass_filter_v < VBP[4]):
                        # partial real power curtailment
                        d = -solar_irr / (VBP[4] - VBP[3])
                        pk = d * (VBP[4]-low_pass_filter_v)
                        qk = (Sbar ** 2 - pk ** 2) ** (1 / 2)
                elif (low_pass_filter_v >= VBP[4]):
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

        # inject to node
        k.node.nodes[node_id]['PQ_injection']['P'] += self.p_out[1]
        k.node.nodes[node_id]['PQ_injection']['Q'] += self.q_out[1]

    def reset(self):
        """See parent class."""
        self.solar_generation = None
        additional_params = self.init_params

        if "default_control_setting" in additional_params:
            self.control_setting = additional_params["default_control_setting"]
        else:
            self.control_setting = DEFAULT_CONTROL_SETTING

        if "low_pass_filter_measure" in additional_params:
            self.low_pass_filter_measure = additional_params["low_pass_filter_measure"]
        else:
            self.low_pass_filter_measure = 1

        if "low_pass_filter_output" in additional_params:
            self.low_pass_filter_output = additional_params["low_pass_filter_output"]
        else:
            self.low_pass_filter_output = 0.1

        if "delta_t" in additional_params:
            self.delta_t = additional_params["delta_t"]
        else:
            self.delta_t = 1

        if "solar_min_value" in additional_params:
            self.solar_min_value = additional_params["solar_min_value"]
        else:
            self.solar_min_value = 5

        self.p_set = np.zeros(2)
        self.q_set = np.zeros(2)
        self.p_out = np.zeros(2)
        self.q_out = np.zeros(2)
        self.low_pass_filter_v = np.zeros(2)

        self.psi = 0
        self.epsilon = 0
        self.y = 0

    def set_control_setting(self, control_setting):
        """See parent class."""
        self.control_setting = control_setting
