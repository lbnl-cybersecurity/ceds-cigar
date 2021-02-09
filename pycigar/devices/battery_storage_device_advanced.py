from collections import deque

import numpy as np

import pycigar.utils.signal_processing as signal_processing
from pycigar.devices.base_device import BaseDevice

from pycigar.utils.logging import logger

DEFAULT_CONTROL_SETTING = 'auto_minmax_cycle'
step_buffer = 4


class BatteryStorageDeviceAdvanced(BaseDevice):

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
        print(self.device_id)
        print(self.current_capacity)

        self.max_charge_power = self.additional_params.get('max_charge_power', 1.00)*1000 # [kW --> W]

        self.max_discharge_power = self.additional_params.get('max_discharge_power', 1.25)*1000 # [kW --> W]

        self.max_ramp_rate = self.additional_params.get('max_ramp_rate', 0.015)*1000 # [kW/s --> W/s]

        print(self.max_ramp_rate)

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
                self.current_capacity = self.current_capacity
            if self.current_capacity <= 0:
                self.current_capacity = self.current_capacity

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
                self.current_capacity = self.current_capacity
            if self.current_capacity <= 0:
                self.current_capacity = self.current_capacity

        # if env.k.time % self.print_interval == 0:
        #     print('Device: ' + self.device_id)
        #     print('p_out_setpoint: ' + str(self.custom_control_setting['p_out']*1000))
        #     print('p_out_rectified: ' + str(self.p_out))
        #     print('')
            

        self.SOC = self.current_capacity/self.total_capacity

        # if 'pout' in self.custom_control_setting:
        #     print(self.custom_control_setting['pout'])

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

        if hasattr(self, 'node_id'):
            Logger.log_single(self.device_id, 'node', self.node_id)
