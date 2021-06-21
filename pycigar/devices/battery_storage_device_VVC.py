from collections import deque

import numpy as np

import pycigar.utils.signal_processing as signal_processing
from pycigar.devices.base_device import BaseDevice

from pycigar.utils.logging import logger

DEFAULT_CONTROL_SETTING = 'auto_minmax_cycle'
step_buffer = 4


class BatteryStorageDeviceVVC(BaseDevice):

    def __init__(self, device_id, additional_params, is_disable_log=False):
        """Instantiate a Storage device."""
        BaseDevice.__init__(
            self,
            device_id,
            additional_params
        )
        self.additional_params = additional_params

        self.control_setting = self.additional_params.get('default_control_setting', DEFAULT_CONTROL_SETTING)

        Logger = logger()

        if 'init_control_settings' in Logger.custom_metrics:
            logger().custom_metrics['init_control_settings'][device_id] =self.control_setting
        else:
            logger().custom_metrics['init_control_settings'] = {device_id: self.control_setting}

        self.p_con = deque([0, 0],maxlen=2)
        self.p_in = deque([0, 0],maxlen=2)
        self.p_out = deque([0, 0],maxlen=2)

        self.p_con = 0 # [W]
        self.p_in = 0 # [W]
        self.p_out = 0 # [W]
        self.total_capacity = self.additional_params.get('total_capacity', 10)*1000*3600 # [kWh --> J]
        self.current_capacity = self.additional_params.get('current_capacity', 10)*1000*3600 # [kWh --> J]
        

        self.max_charge_power = self.additional_params.get('max_charge_power', 1.00)*1000 # [kW --> W]
    
        self.max_discharge_power = self.additional_params.get('max_discharge_power', 1.25)*1000 # [kW --> W]

        self.max_ramp_rate = self.additional_params.get('max_ramp_rate', 0.015)*1000 # [kW/s --> W/s]

        self.s_max = np.max([self.max_discharge_power, self.max_charge_power])*1.1

        self.max_SOC = self.additional_params.get('max_SOC', 1.0)
        self.min_SOC = self.additional_params.get('min_SOC', 0.2)



        self.SOC = self.current_capacity/self.total_capacity

        # Lowpass filter for voltage
        self.Ts = 1 # [s]
        self.fl = 0.25 # [Hz]
        self.lp1s, temp = signal_processing.butterworth_lowpass(1, 2 * np.pi * 1 * self.fl)
        self.lp1z = signal_processing.c2dbilinear(self.lp1s, self.Ts)
        self.Vlp = deque([0]*self.lp1z.shape[1],maxlen=self.lp1z.shape[1])

        # Lowpass filter for power
        # self.Ts = 1
        # self.fl = 0.25
        # lp1s, temp = signal_processing.butterworth_lowpass(1, 2 * np.pi * 1 * self.fl)
        # self.lp1s = lp1s
        # self.lp1z = signal_processing.c2dbilinear(self.lp1s, self.Ts)
        # self.Vlp = deque([0]*self.lp1z.shape[1],maxlen=self.lp1z.shape[1])

        self.custom_control_setting = {}

        #print('Initialize: ' + str(self.device_id))
        #print('current capacity: ' + str(self.current_capacity))
        #print('SOC: ' + str(self.SOC))
        #print('')
        ##################################################
        ##################################################
        self.VVC = np.array([0.98, 1.01, 1.02, 1.05, 1.07]) - 0.03
        self.lpf_high_pass_filter = 1
        self.lpf_low_pass_filter = 0.1
        self.low_pass_filter_v = deque([0] * 2, maxlen=2)
        self.q_set = deque([0] * 2, maxlen=2)
        self.q_out = deque([0] * 2, maxlen=2)
        self.low_pass_filter_measure = 1.2#self.low_pass_filter_measure_std * np.random.randn() + self.low_pass_filter_measure_mean
        self.low_pass_filter_output = 0.1#self.low_pass_filter_output_std * np.random.randn() + self.low_pass_filter_output_mean


    def update(self, k):
        node_id = k.device.get_node_connected_to(self.device_id)
        self.node_id = node_id

        if k.time > 1:
            vk = np.abs(k.node.nodes[node_id]['voltage'][k.time - 1])
            vkm1 = np.abs(k.node.nodes[node_id]['voltage'][k.time - 2])

            self.v_meas_k = vk
            self.v_meas_km1 = vkm1

            if self.lp1z.shape[1] <= 2:
                self.Vlp[-1] = 1/self.lp1z[1,-1]*(np.sum(self.lp1z[1,1]*self.Vlp[-1]) + \
                                              np.sum(self.lp1z[0,:]*[self.v_meas_km1, self.v_meas_k]))
            if self.lp1z.shape[1] >= 3:
                self.Vlp[-1] = 1/self.lp1z[1,-1]*(np.sum(self.lp1z[1,0:-1]*self.Vlp[0:-1]) + \
                                              np.sum(self.lp1z[0,:]*[self.v_meas_km1, self.v_meas_k]))

        if self.control_setting == 'auto_minmax_cycle':
            if self.SOC <= 0.2:
                self.auto_minmax_cycle_mode = 'charge'
                self.current_capacity = self.current_capacity + self.Ts*self.p_in
            if self.SOC >= 0.8:
                self.auto_minmax_cycle_mode = 'discharge'
                self.current_capacity = self.current_capacity - self.Ts*self.p_out
            if self.current_capacity <= 0:
                self.current_capacity = 0
            if self.current_capacity >= self.total_capacity:
                self.current_capacity = self.total_capacity
            self.SOC = self.current_capacity/self.total_capacity

        if self.control_setting == 'external':
            self.current_capacity = self.current_capacity + self.Ts*self.p_con
            if self.current_capacity >= self.total_capacity:
                self.current_capacity = self.total_capacity
            if self.current_capacity <= 0:
                self.current_capacity = 0

        ##### Make p_in, p_out deque size 2?

        if self.control_setting == 'standby':
            self.p_in = 0
            self.p_out = 0

        if self.control_setting == 'charge':
            # self.p_in = 0
            # self.p_in = 150000
            # self.p_in = self.max_charge_power
            self.p_out = 0 # [W]

            if 'p_in' in self.custom_control_setting:

                if self.custom_control_setting['p_in']*1000 >= self.p_in + self.Ts*self.max_ramp_rate: #[W]
                    self.p_in = self.p_in + self.Ts*self.max_ramp_rate # [W]
                else:
                    self.p_in = self.custom_control_setting['p_in']*1000 # [kW --> W]

            if self.current_capacity >= self.max_SOC*self.total_capacity: # [J]

                self.p_in = 0 #[W]
                self.current_capacity = self.max_SOC*self.total_capacity # [J]

            elif self.current_capacity + self.Ts*self.p_in >= self.max_SOC*self.total_capacity: # [J]

                self.p_in = 1/self.Ts*(self.max_SOC*self.total_capacity - self.current_capacity) # [W]
                self.current_capacity = self.max_SOC*self.total_capacity # [J]

            else:

                self.current_capacity = self.current_capacity + self.Ts*self.p_in # [J]

            self.current_capacity = self.current_capacity + self.Ts*self.p_in
            if self.current_capacity <= 0:
                self.current_capacity = 0
            if self.current_capacity >= self.total_capacity:
                self.current_capacity = self.total_capacity
            self.SOC = self.current_capacity/self.total_capacity
            k.node.nodes[node_id]['PQ_injection']['P'] += 1*self.p_in/1000 # [W --> kW]

        if self.control_setting == 'discharge':
            self.p_in = 0
            # self.p_out = 150000
            # self.p_out = 200000
            # self.p_out = self.max_discharge_power

            if 'p_out' in self.custom_control_setting:                
                
                if self.custom_control_setting['p_out']*1000 >= self.p_out + self.Ts*self.max_ramp_rate: # [W]
                    self.p_out = self.p_out + self.Ts*self.max_ramp_rate # [W]
                else:
                    self.p_out = self.custom_control_setting['p_out']*1000 # [kW --> W]

                

            if self.current_capacity <= self.min_SOC*self.total_capacity:

                self.p_out = 0 # [W]
                self.current_capacity = self.min_SOC*self.total_capacity # [J]

            elif self.current_capacity - self.Ts*self.p_out < self.min_SOC*self.total_capacity:

                self.p_out = 1/self.Ts*(self.current_capacity - self.min_SOC*self.total_capacity) # [W]
                self.current_capacity = self.min_SOC*self.total_capacity # [J]

            else:

                self.current_capacity = self.current_capacity - self.Ts*self.p_out # [J]

            self.SOC = self.current_capacity/self.total_capacity
            k.node.nodes[node_id]['PQ_injection']['P'] += -1*self.p_out/1e3 # [W --> kW]

        if self.control_setting == 'voltwatt':
            if self.current_capacity >= self.total_capacity:
                self.current_capacity = self.total_capacity
            if self.current_capacity <= 0:
                self.current_capacity = 0

        # if self.control_setting == 'peak_shaving':
        #     self.p_in = 0
        #     self.p_out = 0


        self.SOC = self.current_capacity/self.total_capacity

        # if 'pout' in self.custom_control_setting:
        #     print(self.custom_control_setting['pout'])

        #########################################################################
        #########################################################################
        T = 1
        lpf_m = self.low_pass_filter_measure
        lpf_o = self.low_pass_filter_output
        VBP = self.VVC
        qk = 0

        if k.time > 1:
            vk = abs(k.node.nodes[self.node_id]['voltage'][k.time - 1])
            vkm1 = abs(k.node.nodes[self.node_id]['voltage'][k.time - 2])
            low_pass_filter_v = (T * lpf_m * (vk + vkm1) - (T * lpf_m - 2) * (self.low_pass_filter_v[1])) / (2 + T * lpf_m)

            # compute p_set and q_set
            print(self.current_capacity, end = " ")
            print(self.control_setting, end = " ")
            print(self.control_setting, end = " ")
            if self.current_capacity >= 0:
                q_avail = (self.s_max ** 2 - self.p_in ** 2  - self.p_out ** 2) ** (1 / 2)
                if low_pass_filter_v <= VBP[4]:
                    # no curtailment
                    print('all:{},t:{},p_in:{},p_out:{},v:{},VBP0: {}'.format(self.total_capacity/1000, self.current_capacity/1000, self.p_in/1000, self.p_out/1000, low_pass_filter_v, VBP[0]), end="/")
                    # determine VAR support
                    if low_pass_filter_v <= VBP[0]:
                        # inject all available var
                        qk = -q_avail
                        print('case 1', end=" ")
                    elif VBP[0] < low_pass_filter_v <= VBP[1]:
                        # partial VAR injection
                        c = q_avail / (VBP[1] - VBP[0])
                        qk = c * (low_pass_filter_v - VBP[1])
                        print('case 2', end=" ")
                    elif VBP[1] < low_pass_filter_v <= VBP[2]:
                        # No var support
                        qk = 0
                        print('case 3', end=" ")
                    elif VBP[2] < low_pass_filter_v < VBP[3]:
                        # partial Var consumption
                        c = q_avail / (VBP[3] - VBP[2])
                        qk = c * (low_pass_filter_v - VBP[2])
                        print('case 4', end=" ")
                    elif VBP[3] < low_pass_filter_v < VBP[4]:
                        # partial real power curtailment
                        qk = (self.s_max ** 2 - self.p_in ** 2  - self.p_out ** 2) ** (1 / 2)
                        print('case 5', end=" ")
                elif low_pass_filter_v >= VBP[4]:
                    # full real power curtailment for VAR support
                    qk = q_avail
                    print('case 6', end=" ")
            self.q_set.append(qk/1000) # [W --> kW]
            self.q_out.append(
                (T * lpf_o * (self.q_set[1] + self.q_set[0]) - (T * lpf_o - 2) * (self.q_out[1])) / (2 + T * lpf_o)
            )

            self.low_pass_filter_v.append(low_pass_filter_v)

        # import old V to x
        # inject to node
        #k.node.nodes[self.node_id]['PQ_injection']['Q'] += self.q_out[1]
        print(k.time, self.node_id, self.q_out[1])
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
        Logger = logger()
        Logger.log(self.device_id, 'control_setting', self.control_setting)
        Logger.log(self.device_id, 'current_capacity', self.current_capacity)
        Logger.log(self.device_id, 'SOC', self.SOC)
        Logger.log(self.device_id, 'p_con', self.p_con)
        Logger.log(self.device_id, 'p_in', self.p_in)
        Logger.log(self.device_id, 'p_out', -self.p_out)

        Logger.log(self.device_id, 'q_out', self.q_out[1])
        Logger.log(self.device_id, 'q_set', self.q_set[1])

        if hasattr(self, 'node_id'):
            Logger.log_single(self.device_id, 'node', self.node_id)