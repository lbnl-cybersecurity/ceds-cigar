from pycigar.controllers.base_controller import BaseController

from collections import deque
import numpy as np
import pycigar.utils.signal_processing as signal_processing

from pycigar.utils.logging import logger


class BatteryPeakShavingControllerCent(BaseController):
    """Fixed controller is the controller that do nothing.
    It only returns the 'default_control_setting' value when being called.
    Attributes
    ----------
    additional_params : dict
        The parameters of the controller
    """

    def __init__(self, device_id, additional_params, controller_id):
        """Instantiate an fixed Controller."""
        BaseController.__init__(
            self,            
            device_id, # devices controlled
            controller_id # this controller
        )
        self.additional_params = additional_params

        print(self.controller_id)
        print(self.device_id)

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

        self.active_power_error = deque([0, 0],maxlen=2)
        self.reactive_power_error = deque([0, 0],maxlen=2)
        self.apparent_power_error = deque([0, 0],maxlen=2)

        self.active_power_error_der = deque([0, 0],maxlen=2)
        self.reactive_power_error_der = deque([0, 0],maxlen=2)
        self.apparent_power_error_der = deque([0, 0],maxlen=2)

        self.active_power_error_int = deque([0, 0],maxlen=2)
        self.reactive_power_error_int = deque([0, 0],maxlen=2)
        self.apparent_power_error_int = deque([0, 0],maxlen=2)

        self.p_set = deque([0, 0],maxlen=2)

        self.p_in = deque([0, 0],maxlen=2)
        self.p_out = deque([0, 0],maxlen=2)

        # self.eta = 0.01
        self.eta = self.additional_params.get('eta', 0.05)

        self.K_P = self.additional_params.get('K_P', 0.1)
        self.K_I = self.additional_params.get('K_I', 0.01)
        self.K_D = self.additional_params.get('K_D', 0.01)
        

        self.print_interval = 1

    def get_action(self, env):

        result = {}

        if env.k.time == 0 or env.k.time == 51:

            print('Time: ' + str(env.k.time))
            print('Initialize: ' + str(self.device_id))

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

            self.active_power_error.append(self.P_target - self.measured_active_power_lpf[-1])
            self.reactive_power_error.append(self.P_target - self.measured_reactive_power_lpf[-1])
            self.apparent_power_error.append(self.P_target - self.measured_apparent_power_lpf[-1])

            self.active_power_error_der.append(1/self.Ts*(self.active_power_error[-1] - self.active_power_error[-2]))
            self.reactive_power_error_der.append(1/self.Ts*(self.reactive_power_error[-1] - self.reactive_power_error[-2]))
            self.apparent_power_error_der.append(1/self.Ts*(self.apparent_power_error[-1] - self.apparent_power_error[-2]))

            self.active_power_error_int.append(self.active_power_error_int[-2] + self.Ts*self.active_power_error[-2])
            self.reactive_power_error_int.append(self.reactive_power_error_int[-2] + self.Ts*self.reactive_power_error[-2])
            self.apparent_power_error_int.append(self.apparent_power_error_int[-2] + self.Ts*self.apparent_power_error[-2])

            self.p_set_temp = self.K_P*self.active_power_error[-1] + self.K_I*self.active_power_error_int[-1] + self.K_D*self.active_power_error_der[-1]

            # self.p_set.append(self.p_set[-2] + self.eta * (self.P_target - self.measured_active_power_lpf[-2]))

            self.p_set.append(self.p_set_temp)

            # if self.p_set <= 0:
            #     if self.p_set <= env.k.devices[self.device_id].SOC

            # self.p_set.append((1 - self.Ts*self.eta)*self.p_set[-1] - self.Ts*self.eta*(self.P_target - self.measured_active_power_lpf[-2]))

            result = {}

            max_charge_power_list = []
            max_discharge_power_list = []

            for device in self.device_id:

                max_charge_power_list.append(env.k.device.devices[device]['device'].max_charge_power/1e3)
                max_discharge_power_list.append(env.k.device.devices[device]['device'].max_discharge_power/1e3)

            for device in self.device_id:

                # print(self.p_set[-1])

                if self.p_set[-1] >= 0:

                    # if self.p_set[-1] >= np.abs(self.measured_active_power_lpf[-2] - self.P_target):
                    #     self.p_set[-1] = np.abs(self.measured_active_power_lpf[-2] - self.P_target)

                    if self.p_set[-1] >= max(max_charge_power_list):
                        self.p_set[-1] = max(max_charge_power_list)

                    if self.p_set[-1] >= env.k.device.devices[device]['device'].max_charge_power/1e3:
                        # self.p_set[-1] = env.k.device.devices[device]['device'].max_charge_power/1e3
                        result[device] = ('charge',  {'p_in': env.k.device.devices[device]['device'].max_charge_power/1e3})
                    else:
                        result[device] = ('charge',  {'p_in': 1*self.p_set[-1]})
                    # if self.p_set[-1] >= (self.P_target - self.measured_active_power_lpf[-2]):
                    #     self.p_set[-1] = (self.P_target - self.measured_active_power_lpf[-2])

                    # self.control_setting.append('charge')
                    # self.p_in.append(self.p_set[-1])
                    # self.p_out.append(0)
                    # self.custom_control_setting = {'p_in': 1*self.p_in[-1]}

                elif self.p_set[-1] <= 0:

                    # if self.p_set[-1] <= np.abs(self.measured_active_power_lpf[-2] - self.P_target):
                    #     self.p_set[-1] = -np.abs(self.measured_active_power_lpf[-2] - self.P_target)

                    if self.p_set[-1] <= -max(max_discharge_power_list):
                        self.p_set[-1] = -max(max_discharge_power_list)

                    if self.p_set[-1] <= -env.k.device.devices[device]['device'].max_discharge_power/1e3:
                        # self.p_out[-1] = -env.k.device.devices[device]['device'].max_discharge_power/1e3
                        result[device] = ('discharge',  {'p_out': env.k.device.devices[device]['device'].max_discharge_power/1e3})
                    else:
                        # self.p_out.append(self.p_set[-1])
                        result[device] = ('discharge',  {'p_out': -1*self.p_set[-1]})

                    # if self.p_set[-1] <= (self.P_target - self.measured_active_power_lpf[-2]):
                    #     self.p_set[-1] = (self.P_target - self.measured_active_power_lpf[-2])
                
                    # self.control_setting.append('discharge')
                    # self.p_in.append(0)
                    # # self.p_out.append(-self.p_set[-1])
                    # self.custom_control_setting = {'p_out': 1*self.p_out[-1]}

                    # result[device] = ('discharge',  {'p_out': 1*self.p_out[-1]})

            # return result

            if env.k.time % self.print_interval == 0:
                print('Time: ' + str(env.k.time))
                print('Controller: ' + self.controller_id)

                for device in self.device_id:
                    print('Device: ' + str(device))
                    print('Battery SOC: ' + str(env.k.device.devices[device]['device'].SOC))

                # print('Measured active power [kW]: ' + str(self.measured_active_power[-1]))
                # print('Measured reactive power [kVAr]: ' + str(self.measured_reactive_power[-1]))
                # print('Measured apparent power [kVA]: ' + str(self.measured_apparent_power[-1]))

                print('Measured active power lpf [kW]: ' + str(self.measured_active_power_lpf[-1]))
                print('Measured reactive power lpf [kVAr]: ' + str(self.measured_reactive_power_lpf[-1]))
                print('Measured apparent power lpf [kVA]: ' + str(self.measured_apparent_power_lpf[-1]))

                # if self.control_setting[-2] == 'charge':
                #     print('Charge')
                #     print('Active Power Control k-1 [kW]: ' + str(self.p_in[-2]))
                # if self.control_setting[-1] == 'discharge':
                #     print('Discharge')
                #     print('Active Power Control k-1 [kW]: ' + str(self.p_out[-2]))

                # print('Load active power [kW]: ' + str(self.load_active_power[-1]))
                # print('Load reactive power [kVAr]: ' + str(self.load_reactive_power[-1]))
                # print('Load apparent power [kVA]: ' + str(self.load_apparent_power[-1]))
                # print('')

                print(result)
                for device in self.device_id:
                    print('Device: ' + str(device))
                    print(result[device])

                # if self.control_setting[-1] == 'charge':
                #     print('Discharge')
                #     # print('Discharge power non rectified [kW]: ' + str(self.p_set[-1]))
                #     print('Discharge power [kW]: ' + str(self.p_in[-1]))

                # if self.control_setting[-1] == 'discharge':
                #     print('Discharge')
                #     # print('Discharge power non rectified [kW]: ' + str(self.p_set[-1]))
                #     print('Discharge power [kW]: ' + str(self.p_out[-1]))            
                

                print('')

        self.log()

        return result

        # return self.control_setting[-1], self. custom_control_setting

        


    def reset(self):
        """See parent class."""
        pass

    def log(self):

        # log history
        Logger = logger()
        Logger.log(self.controller_id, 'control_setting', self.control_setting[-1])
        Logger.log(self.controller_id, 'control_setting', self.custom_control_setting)

        Logger.log(self.controller_id, 'measured_active_power', self.measured_active_power[-1])
        Logger.log(self.controller_id, 'measured_reactive_power', self.measured_reactive_power[-1])
        Logger.log(self.controller_id, 'measured_apparent_power', self.measured_apparent_power[-1])

        Logger.log(self.controller_id, 'measured_active_power_lpf', self.measured_active_power_lpf[-1])
        Logger.log(self.controller_id, 'measured_reactive_power_lpf', self.measured_reactive_power_lpf[-1])
        Logger.log(self.controller_id, 'measured_apparent_power_lpf', self.measured_apparent_power_lpf[-1])

        Logger.log(self.controller_id, 'load_active_power', self.load_active_power[-1])
        Logger.log(self.controller_id, 'load_reactive_power', self.load_reactive_power[-1])
        Logger.log(self.controller_id, 'load_apparent_power', self.load_apparent_power[-1])

        Logger.log(self.controller_id, 'p_target', self.P_target)

        Logger.log(self.controller_id, 'p_set', self.p_set[-1])
        Logger.log(self.controller_id, 'p_in', self.p_in[-1])
        Logger.log(self.controller_id, 'p_out', self.p_out[-1])
        

        # Logger.log(self.device_id, 'current_capacity', self.current_capacity)
        # Logger.log(self.device_id, 'SOC', self.SOC)
        # Logger.log(self.device_id, 'p_con', self.p_con)
        # Logger.log(self.device_id, 'p_in', self.p_in)
        # Logger.log(self.device_id, 'p_out', self.p_out)

        # if hasattr(self, 'node_id'):
        #     Logger.log_single(self.device_id, 'node', self.node_id)