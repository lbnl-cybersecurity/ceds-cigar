from pycigar.controllers.base_controller import BaseController

from collections import deque
import numpy as np
import pycigar.utils.signal_processing as signal_processing

from pycigar.utils.logging import logger


class BatteryPeakShavingControllerDist(BaseController):
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
            device_id,
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

        # self.P_target = 725
        self.P_target = self.additional_params.get('active_power_target', 800)

        self.min_active_power = -500
        self.max_active_power = 900

        self.max_apparent_power = 1000

        # Lowpass filter for power
        self.Ts = 1
        # self.fl = 0.010
        self.fl = self.additional_params.get('lowpass_filter_frequency', 0.010)
        lp1s, temp = signal_processing.butterworth_lowpass(1, 2 * np.pi * 1 * self.fl)
        self.lp1s = lp1s
        self.lp1z = signal_processing.c2dbilinear(self.lp1s, self.Ts)
        self.Plp = deque([0]*self.lp1z.shape[1],maxlen=self.lp1z.shape[1])

        # print(self.lp1s)
        # print(self.lp1z)

        self.measured_active_power = deque([0, 0],maxlen=2)
        self.measured_reactive_power = deque([0, 0],maxlen=2)
        self.measured_apparent_power = deque([0, 0],maxlen=2)

        self.measured_active_power_lpf = deque([0, 0],maxlen=2)
        self.measured_reactive_power_lpf = deque([0, 0],maxlen=2)
        self.measured_apparent_power_lpf = deque([0, 0],maxlen=2)

        self.load_active_power = deque([0, 0],maxlen=2)
        self.load_reactive_power = deque([0, 0],maxlen=2)
        self.load_apparent_power = deque([0, 0],maxlen=2)

        self.load_active_power_lpf = deque([0, 0],maxlen=2)
        self.load_reactive_power_lpf = deque([0, 0],maxlen=2)
        self.load_apparent_power_lpf = deque([0, 0],maxlen=2)
        
        self.control_setting = deque(['standby', 'standby'],maxlen=2)
        self.custom_control_setting = {}

        self.p_set = deque([0, 0],maxlen=2)

        self.p_in = deque([0, 0],maxlen=2)
        self.p_out = deque([0, 0],maxlen=2)

        # self.eta = 0.01
        self.eta = self.additional_params.get('eta', 0.02)
        

        self.print_interval = 1

    def get_action(self, env):

        if env.k.time == 0 or env.k.time == 51:

            print('Time: ' + str(env.k.time))
            print('Initialize: ' + self.device_id)

            for k1 in range(0,2):               

                self.measured_active_power.append(-env.k.kernel_api.get_total_power()[0])
                self.measured_reactive_power.append(-env.k.kernel_api.get_total_power()[1])
                self.measured_apparent_power.append((self.measured_active_power[-1]**2 + self.measured_reactive_power[-1]**2)**(1/2))

                self.measured_active_power_lpf.append(-env.k.kernel_api.get_total_power()[0])
                self.measured_reactive_power_lpf.append(-env.k.kernel_api.get_total_power()[1])
                self.measured_apparent_power_lpf.append((self.measured_active_power[-1]**2 + self.measured_reactive_power[-1]**2)**(1/2))

                self.load_active_power.append(-env.k.kernel_api.get_total_power()[0])
                self.load_reactive_power.append(-env.k.kernel_api.get_total_power()[1])
                self.load_apparent_power.append((self.load_active_power[-1]**2 + self.load_reactive_power[-1]**2)**(1/2))

                self.load_active_power_lpf.append(-env.k.kernel_api.get_total_power()[0])
                self.load_reactive_power_lpf.append(-env.k.kernel_api.get_total_power()[1])
                self.load_apparent_power_lpf.append((self.load_active_power_lpf[-1]**2 + self.load_reactive_power_lpf[-1]**2)**(1/2))

        else:

            # Pk = np.abs(k.node.nodes[node_id]['voltage'][k.time - 1])
            # Pkm1 = np.abs(k.node.nodes[node_id]['voltage'][k.time - 2])

            self.measured_active_power.append(-env.k.kernel_api.get_total_power()[0])
            self.measured_reactive_power.append(-env.k.kernel_api.get_total_power()[1])

            # print(total_apparent_power)
            self.measured_apparent_power.append((self.measured_active_power[-1]**2 + self.measured_reactive_power[-1]**2)**(1/2))
            # print(total_apparent_power)

            # if self.lp1z.shape[1] <= 2:
            #     self.Plp[-1] = 1/self.lp1z[1,-1]*(np.sum(self.lp1z[1,1]*self.Plp[-1]) + np.sum(self.lp1z[0,:]*[self.v_meas_km1, self.v_meas_k]))

            ptemp = 1/self.lp1z[1,1]*(-self.lp1z[1,0]*self.measured_active_power_lpf[0] + self.lp1z[0,0]*self.measured_active_power[0] + self.lp1z[0,1]*self.measured_active_power[1])
            self.measured_active_power_lpf.append(ptemp)

            qtemp = 1/self.lp1z[1,1]*(-self.lp1z[1,0]*self.measured_reactive_power_lpf[0] + self.lp1z[0,0]*self.measured_reactive_power[0] + self.lp1z[0,1]*self.measured_reactive_power[1])
            self.measured_reactive_power_lpf.append(qtemp)

            self.measured_apparent_power_lpf.append((self.measured_active_power_lpf[-1]**2 + self.measured_reactive_power_lpf[-1]**2)**(1/2))

            ##################################################
            ##################################################
            ##################################################

            # self.p_set_temp - self.p_set[-1] - self.eta * (self.P_target - self.measured_active_power_lpf[-2])
            # if self.p_set_temp >= 0 and self.p_set_temp >= self.self.measured_active_power_lpf[-2] - self.P_target: 
            #     self.p_set_temp = self.P_target - self.self.measured_active_power_lpf[-2]
            # elif self.p_set_temp <= 0 and self.p_set_temp <= self.P_target - self.self.measured_active_power_lpf[-2]:
            #     self.p_set_temp = self.P_target - self.self.measured_active_power_lpf[-2]

            self.p_set.append(self.p_set[-2] - self.eta * (self.P_target - self.measured_active_power_lpf[-2]))

            # self.p_set.append((1 - self.Ts*self.eta)*self.p_set[-1] - self.Ts*self.eta*(self.P_target - self.measured_active_power_lpf[-2]))

            if self.p_set[-1] <= 0:

                self.control_setting.append('charge')
                self.p_in.append(-self.p_set[-1])
                self.p_out.append(0)
                self.custom_control_setting = {'p_in': 1e3*self.p_in[-1]}

            elif self.p_set[-1] > 0:
                
                self.control_setting.append('discharge')
                self.p_in.append(0)
                self.p_out.append(self.p_set[-1])
                self.custom_control_setting = {'p_out': 1e3*self.p_out[-1]}

            ##################################################
            ##################################################
            ##################################################

            # if self.control_setting[-1] == 'charge':
            #     self.load_active_power.append(self.measured_active_power_lpf[-1] - self.p_in[-2])
            # if self.control_setting[-1] == 'discharge':
            #     self.load_active_power.append(self.measured_active_power_lpf[-1] + self.p_out[-2])

            # self.load_reactive_power.append(self.measured_reactive_power[-1])
            # self.load_apparent_power.append((self.load_active_power[-1]**2 + self.load_reactive_power[-1]**2)**(1/2))

            # if self.load_active_power[-1] >= self.max_active_power:

            #     self.control_setting.append('discharge')

            #     self.p_set.append(self.load_active_power[-1] - self.max_active_power)

            #     self.p_out.append(0*min(self.p_set[-1], env.k.device.devices[self.device_id]['device'].max_discharge/1e3))

            #     self.p_in.append(0)

            #     self.custom_control_setting = {'p_out': 1e3*self.p_out[-1]}

            # elif self.load_active_power[-1] <= self.max_active_power and self.load_active_power[-1] >= 0:

            #     self.control_setting.append('charge')

            #     self.p_set.append(self.max_active_power - self.load_active_power[-1])

            #     self.p_in.append(0*min(self.p_set[-1], env.k.device.devices[self.device_id]['device'].max_charge/1e3))

            #     self.p_out.append(0)

            #     self.custom_control_setting = {'p_in': 1e3*self.p_in[-1]}

            ##################################################
            ##################################################
            ##################################################
            
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
            print('Time: ' + str(env.k.time))
            print('Device: ' + self.device_id)

            print('Measured active power [kW]: ' + str(self.measured_active_power[-1]))
            print('Measured reactive power [kVAr]: ' + str(self.measured_reactive_power[-1]))
            print('Measured apparent power [kVA]: ' + str(self.measured_apparent_power[-1]))

            print('Measured active power lpf [kW]: ' + str(self.measured_active_power_lpf[-1]))
            print('Measured reactive power lpf [kVAr]: ' + str(self.measured_reactive_power_lpf[-1]))
            print('Measured apparent power lpf [kVA]: ' + str(self.measured_apparent_power_lpf[-1]))

            if self.control_setting[-2] == 'charge':
                print('Charge')
                print('Active Power Control k-1 [kW]: ' + str(self.p_in[-2]))
            if self.control_setting[-1] == 'discharge':
                print('Discharge')
                print('Active Power Control k-1 [kW]: ' + str(self.p_out[-2]))

            print('Load active power [kW]: ' + str(self.load_active_power[-1]))
            print('Load reactive power [kVAr]: ' + str(self.load_reactive_power[-1]))
            print('Load apparent power [kVA]: ' + str(self.load_apparent_power[-1]))
            # print('')

            if self.control_setting[-1] == 'charge':
                print('Discharge')
                print('Discharge power non rectified [kW]: ' + str(self.p_set[-1]))
                print('Discharge power [kW]: ' + str(self.p_in[-1]))

            if self.control_setting[-1] == 'discharge':
                print('Discharge')
                print('Discharge power non rectified [kW]: ' + str(self.p_set[-1]))
                print('Discharge power [kW]: ' + str(self.p_out[-1]))            
            

            print('')

            self.log()

        return self.control_setting[-1], self. custom_control_setting

        


    def reset(self):
        """See parent class."""
        pass

    def log(self):

        # log history
        Logger = logger()
        Logger.log(self.device_id + '_psc', 'control_setting', self.control_setting[-1])
        Logger.log(self.device_id + '_psc', 'control_setting', self.custom_control_setting)

        Logger.log(self.device_id + '_psc', 'measured_active_power', self.measured_active_power[-1])
        Logger.log(self.device_id + '_psc', 'measured_reactive_power', self.measured_reactive_power[-1])
        Logger.log(self.device_id + '_psc', 'measured_apparent_power', self.measured_apparent_power[-1])

        Logger.log(self.device_id + '_psc', 'measured_active_power_lpf', self.measured_active_power_lpf[-1])
        Logger.log(self.device_id + '_psc', 'measured_reactive_power_lpf', self.measured_reactive_power_lpf[-1])
        Logger.log(self.device_id + '_psc', 'measured_apparent_power_lpf', self.measured_apparent_power_lpf[-1])

        Logger.log(self.device_id + '_psc', 'load_active_power', self.load_active_power[-1])
        Logger.log(self.device_id + '_psc', 'load_reactive_power', self.load_reactive_power[-1])
        Logger.log(self.device_id + '_psc', 'load_apparent_power', self.load_apparent_power[-1])

        Logger.log(self.device_id + '_psc', 'p_target', self.P_target)

        Logger.log(self.device_id + '_psc', 'p_set', self.p_set[-1])
        Logger.log(self.device_id + '_psc', 'p_in', self.p_in[-1])
        Logger.log(self.device_id + '_psc', 'p_out', self.p_out[-1])
        

        # Logger.log(self.device_id, 'current_capacity', self.current_capacity)
        # Logger.log(self.device_id, 'SOC', self.SOC)
        # Logger.log(self.device_id, 'p_con', self.p_con)
        # Logger.log(self.device_id, 'p_in', self.p_in)
        # Logger.log(self.device_id, 'p_out', self.p_out)

        # if hasattr(self, 'node_id'):
        #     Logger.log_single(self.device_id, 'node', self.node_id)