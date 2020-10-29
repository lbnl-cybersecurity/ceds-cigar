from pycigar.controllers.base_controller import BaseController

from collections import deque
import numpy as np
import pycigar.utils.signal_processing as signal_processing


class BatteryPeakShavingController(BaseController):
    """Fixed controller is the controller that do nothing.
    It only returns the 'default_control_setting' value when being called.
    Attributes
    ----------
    additional_params : dict
        The parameters of the controller
    """

    def __init__(self, device_id, additional_params):
        """Instantiate an fixed Controller."""
        BaseController.__init__(
            self,
            device_id
        )
        self.additional_params = additional_params

#     def get_action(self, env):
#         """See parent class."""
#         power_reactive_power = env.k.kernel_api.get_total_power()
#         if env.k.device.devices[self.device_id]['device'].SOC <= 0.2:
# #             print('CHARGE')
#             control_setting = 'charge'
#         elif env.k.device.devices[self.device_id]['device'].SOC >= 0.8:
# #             print('DISCHARGE')
#             control_setting = 'discharge'
#         else:
#             control_setting = env.k.device.devices[self.device_id]['device'].control_setting

#         return control_setting

        self.p_in = 0

        self.p_out = 0

        self.min_active_power = -500
        self.max_active_power = 900

        self.max_apparent_power = 1000

        # Lowpass filter for power
        self.Ts = 1
        self.fl = 0.25
        lp1s, temp = signal_processing.butterworth_lowpass(1, 2 * np.pi * 1 * self.fl)
        self.lp1s = lp1s
        self.lp1z = signal_processing.c2dbilinear(self.lp1s, self.Ts)
        self.Plp = deque([0]*self.lp1z.shape[1],maxlen=self.lp1z.shape[1])

        self.measured_active_power = 0
        self.measured_reactive_power = 0
        self.measured_apparent_power = 0

        self.load_active_power = 0
        self.load_reactive_power = 0
        self.load_apparent_power = 0

        self.control_setting = 'charge'
        self.custom_control_setting = {}

        self.print_interval = 1

    def get_action(self, env):

        # Pk = np.abs(k.node.nodes[node_id]['voltage'][k.time - 1])
        # Pkm1 = np.abs(k.node.nodes[node_id]['voltage'][k.time - 2])

        self.measured_active_power = -env.k.kernel_api.get_total_power()[0]
        self.measured_reactive_power = -env.k.kernel_api.get_total_power()[1]

        # print(total_apparent_power)
        self.measured_apparent_power = (self.measured_active_power**2 + self.measured_reactive_power**2)**(1/2)
        # print(total_apparent_power)

        # if self.lp1z.shape[1] <= 2:
        #     self.Plp[-1] = 1/self.lp1z[1,-1]*(np.sum(self.lp1z[1,1]*self.Plp[-1]) + np.sum(self.lp1z[0,:]*[self.v_meas_km1, self.v_meas_k]))

        if self.control_setting == 'charge':
            self.load_active_power = self.measured_active_power - self.p_in
        if self.control_setting == 'discharge':
            self.load_active_power = self.measured_active_power + self.p_out

        self.load_reactive_power = self.measured_reactive_power
        self.load_apparent_power = (self.load_active_power**2 + self.load_reactive_power**2)**(1/2)

        if env.k.time % self.print_interval == 0:
            print('Time: ' + str(env.k.time))

            print('Measured active power [kW]: ' + str(self.measured_active_power))
            print('Measured reactive power [kVAr]: ' + str(self.measured_reactive_power))
            print('Measured apparent power [kVA]: ' + str(self.measured_apparent_power))

            print('Active Power Control k-1 [kW]: ' + str(self.p_out))

            print('Load active power [kW]: ' + str(self.load_active_power))
            print('Load reactive power [kVAr]: ' + str(self.load_reactive_power))
            print('Load apparent power [kVA]: ' + str(self.load_apparent_power))
            # print('')

        if env.k.time == 0 or env.k.time == 51:
            pass
        else:
            pass

            

        if self.load_active_power >= self.max_active_power:

            self.control_setting = 'discharge'

            self.p_set = self.load_active_power - self.max_active_power

            self.p_out = min(self.p_set, env.k.device.devices[self.device_id]['device'].max_discharge/1e3)

            self.custom_control_setting = {'p_out': 1e3*self.p_out}

            if env.k.time % self.print_interval == 0:
                print('Discharge')
                print('Discharge power non rectified [kW]: ' + str(self.p_set))
                print('Discharge power [kW]: ' + str(self.p_out))

        elif self.load_active_power <= self.max_active_power and self.load_active_power >= 0:

            self.control_setting = 'charge'

            self.p_set = self.max_active_power - self.load_active_power

            self.p_in = min(self.p_set, env.k.device.devices[self.device_id]['device'].max_charge/1e3)

            self.custom_control_setting = {'p_in': 1e3*self.p_in}

            if env.k.time % self.print_interval == 0:
                print('Charge')
                print('Charge power non rectified [kW]: ' + str(self.p_set))
                print('Charge power [kW]: ' + str(self.p_in))

        
        # if self.total_apparent_power >= self.max_apparent_power:

        #     if self.total_active_power >= 0:

        #         self.control_setting = 'discharge'

        #         self.p_set = (self.max_apparent_power - (self.max_apparent_power**2 - self.total_reactive_power**2)**(1/2))

        #         self.p_out = min(self.p_set, env.k.device.devices[self.device_id]['device'].max_discharge/1e3)

        #         self.custom_control_setting = {'p_out': 1e3*self.p_out}

        #         if env.k.time % self.print_interval == 0:
        #             print('Discharge')
        #             print('Discharge power non rectified [kW]: ' + str(self.p_set))
        #             print('Discharge power [kW]: ' + str(self.p_out))

        #     if self.total_active_power <= 0:
                
        #         self.control_setting = 'charge'

        #         self.p_set = 1e3*(self.max_apparent_power - (self.max_apparent_power**2 - self.total_reactive_power**2)**(1/2))

        #         self.p_in = min(self.p_set, env.k.device.devices[self.device_id]['device'].max_charge/1e3)

        #         self.custom_control_setting = {'p_in': 1e3*self.p_in}

        #         if env.k.time % self.print_interval == 0:
        #                 print('Charge')
        #                 print('Charge power non rectified [kW]: ' + str(self.p_set))
        #                 print('Charge power [kW]: ' + str(self.p_in))
            
            

        if env.k.time % self.print_interval == 0:
            print('')

        return self.control_setting, self. custom_control_setting

        


    def reset(self):
        """See parent class."""
        pass

    def log(self):

        # log history
        Logger = logger()
        Logger.log(self.device_id, 'control_setting', self.control_setting)
        # Logger.log(self.device_id, 'current_capacity', self.current_capacity)
        # Logger.log(self.device_id, 'SOC', self.SOC)
        # Logger.log(self.device_id, 'p_con', self.p_con)
        # Logger.log(self.device_id, 'p_in', self.p_in)
        # Logger.log(self.device_id, 'p_out', self.p_out)

        # if hasattr(self, 'node_id'):
        #     Logger.log_single(self.device_id, 'node', self.node_id)