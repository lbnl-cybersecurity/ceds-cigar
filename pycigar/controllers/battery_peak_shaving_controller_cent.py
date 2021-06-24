from pycigar.controllers.base_controller import BaseController
from pycigar.controllers.rl_controller import RLController
from collections import deque
import numpy as np
import pycigar.utils.signal_processing as signal_processing
import math
from pycigar.utils.logging import logger


class BatteryPeakShavingControllerCent(RLController):
    """Fixed controller is the controller that do nothing.
    It only returns the 'default_control_setting' value when being called.
    Attributes
    ----------
    additional_params : dict
        The parameters of the controller
    """

    def __init__(self, device_id, additional_params, controller_id):
        """Instantiate an fixed Controller."""
        RLController.__init__(
            self,
            device_id=device_id, # devices controlled
            controller_id=controller_id, # this controller
        )
        self.additional_params = additional_params

        # Substation active power target, for target tracking, peak shaving, valley filling
        self.P_target = self.additional_params.get('active_power_target', 1000)
        # Substation reactive power target, for target tracking, peak shaving, valley filling
        self.Q_target = self.additional_params.get('reactive_power_target', 1000)
        # Substation aapparent power target, for target tracking, peak shaving, valley filling
        self.S_target = self.additional_params.get('apaprent_power_target', 1000)


        # Lowpass filter for substation power
        # timestep
        self.Ts = 1
        # lowpass filter frequency
        self.fl = self.additional_params.get('lowpass_filter_frequency', 0.10)
        # lowpass filter array in CT
        lp1s, temp = signal_processing.butterworth_lowpass(1, 2 * np.pi * 1 * self.fl)
        self.lp1s = lp1s
        # lowpass filter array in DT
        self.lp1z = signal_processing.c2dbilinear(self.lp1s, self.Ts)

        # Measured substation active, reactive, apparent power values in deque arrays
        self.measured_active_power = deque([0, 0],maxlen=2)
        self.measured_reactive_power = deque([0, 0],maxlen=2)
        self.measured_apparent_power = deque([0, 0],maxlen=2)

        # LPF substation active, reactive, apparent power values in deque arrays
        self.measured_active_power_lpf = deque([0, 0],maxlen=2)
        self.measured_reactive_power_lpf = deque([0, 0],maxlen=2)
        self.measured_apparent_power_lpf = deque([0, 0],maxlen=2)

        # Load active, reactive, apparent power values in deque arrays
        self.load_active_power = deque([0, 0],maxlen=2)
        self.load_reactive_power = deque([0, 0],maxlen=2)
        self.load_apparent_power = deque([0, 0],maxlen=2)

        # LPF load active, reactive, apparent power values in deque arrays
        self.load_active_power_lpf = deque([0, 0],maxlen=2)
        self.load_reactive_power_lpf = deque([0, 0],maxlen=2)
        self.load_apparent_power_lpf = deque([0, 0],maxlen=2)
        
        # Control setting in deque array
        self.control_setting = deque(['standby', 'standby'],maxlen=2)
        self.custom_control_setting = {}

        # Active, reactive, apparent power error for PID
        self.active_power_error = deque([0, 0],maxlen=2)
        self.reactive_power_error = deque([0, 0],maxlen=2)
        self.apparent_power_error = deque([0, 0],maxlen=2)

        # Derivative of active, reactive, apparent power error for PID
        self.active_power_error_der = deque([0, 0],maxlen=2)
        self.reactive_power_error_der = deque([0, 0],maxlen=2)
        self.apparent_power_error_der = deque([0, 0],maxlen=2)

        # Integral of active, reactive, apparent power error for PID
        self.active_power_error_int = deque([0, 0],maxlen=2)
        self.reactive_power_error_int = deque([0, 0],maxlen=2)
        self.apparent_power_error_int = deque([0, 0],maxlen=2)

        # Active power setpoint for BSD, deque array
        self.p_set = deque([0, 0],maxlen=2)
        # Temporary active power setpoint
        self.p_set_temp = 0

        self.p_in = deque([0, 0],maxlen=2)
        self.p_out = deque([0, 0],maxlen=2)

        # PID gains
        self.K_P = self.additional_params.get('K_P', 0.1)
        self.K_I = self.additional_params.get('K_I', 0.01)
        self.K_D = self.additional_params.get('K_D', 0.01)

        self.control_mode = self.additional_params.get('control_mode', 'standby')

        # Timestep interval for printing
        self.print_interval = 1

    def get_action(self, env):

        power = env.k.kernel_api.get_total_power()
        if math.isinf(power[0]) or math.isinf(power[1]) or math.isnan(power[0]) or math.isnan(power[1]):
            active_power = self.measured_active_power[-1]
            reactive_power = self.measured_reactive_power[-1]
        else:
            active_power = power[0]
            reactive_power = power[1]

        result = {}

        if env.k.time == 0 or env.k.time == 51:
            self.control_mode = 'peak_shaving'
            self.P_target = 3000

        if env.k.time == 2500:
            self.control_mode = 'peak_shaving'
            self.P_target = 3000

        if env.k.time == 6000:
            self.control_mode = 'valley_filling'
            self.P_target = 2600

        # Initialization period for pycigar
        if env.k.time == 0 or env.k.time == 51:

            # Fill deque arrays with same value, and intialized LPFs
            for k1 in range(0,2):

                self.measured_active_power.append(active_power)
                self.measured_reactive_power.append(reactive_power)
                self.measured_apparent_power.append((self.measured_active_power[-1]**2 + self.measured_reactive_power[-1]**2)**(1/2))

                self.measured_active_power_lpf.append(active_power)
                self.measured_reactive_power_lpf.append(reactive_power)
                self.measured_apparent_power_lpf.append((self.measured_active_power[-1]**2 + self.measured_reactive_power[-1]**2)**(1/2))

                self.load_active_power.append(active_power)
                self.load_reactive_power.append(reactive_power)
                self.load_apparent_power.append((self.load_active_power[-1]**2 + self.load_reactive_power[-1]**2)**(1/2))

                self.load_active_power_lpf.append(active_power)
                self.load_reactive_power_lpf.append(reactive_power)
                self.load_apparent_power_lpf.append((self.load_active_power_lpf[-1]**2 + self.load_reactive_power_lpf[-1]**2)**(1/2))

        else:

            # Pk = np.abs(k.node.nodes[node_id]['voltage'][k.time - 1])
            # Pkm1 = np.abs(k.node.nodes[node_id]['voltage'][k.time - 2])

            self.measured_active_power.append(active_power)
            self.measured_reactive_power.append(reactive_power)

            # Measured substation apparent power
            self.measured_apparent_power.append((self.measured_active_power[-1]**2 + self.measured_reactive_power[-1]**2)**(1/2))
            # print(total_apparent_power)

            # LPF substation active power
            ptemp = 1/self.lp1z[1,1]*(-self.lp1z[1,0]*self.measured_active_power_lpf[0] + self.lp1z[0,0]*self.measured_active_power[0] + self.lp1z[0,1]*self.measured_active_power[1])
            self.measured_active_power_lpf.append(ptemp)

            # LPF substation reactive power
            qtemp = 1/self.lp1z[1,1]*(-self.lp1z[1,0]*self.measured_reactive_power_lpf[0] + self.lp1z[0,0]*self.measured_reactive_power[0] + self.lp1z[0,1]*self.measured_reactive_power[1])
            self.measured_reactive_power_lpf.append(qtemp)

            # LPF substation apparent power
            # self.measured_apparent_power_lpf.append((self.measured_active_power_lpf[-1]**2 + self.measured_reactive_power_lpf[-1]**2)**(1/2))
            stemp = 1/self.lp1z[1,1]*(-self.lp1z[1,0]*self.measured_apparent_power_lpf[0] + self.lp1z[0,0]*self.measured_apparent_power[0] + self.lp1z[0,1]*self.measured_apparent_power[1])
            self.measured_apparent_power_lpf.append(stemp)


            ##################################################
            # PID control, based on control mode

            if self.control_mode == 'power_target_tracking':

                # compute reference error
                self.active_power_error.append(self.P_target - self.measured_active_power_lpf[-1])
                self.reactive_power_error.append(self.Q_target - self.measured_reactive_power_lpf[-1])
                self.apparent_power_error.append(self.S_target - self.measured_apparent_power_lpf[-1])

                # compute integral of reference error
                self.active_power_error_int.append(self.active_power_error_int[-2] + self.Ts*self.active_power_error[-2])
                self.reactive_power_error_int.append(self.reactive_power_error_int[-2] + self.Ts*self.reactive_power_error[-2])
                self.apparent_power_error_int.append(self.apparent_power_error_int[-2] + self.Ts*self.apparent_power_error[-2])

                # compute derivative of reference error
                self.active_power_error_der.append(1/self.Ts*(self.active_power_error[-1] - self.active_power_error[-2]))
                self.reactive_power_error_der.append(1/self.Ts*(self.reactive_power_error[-1] - self.reactive_power_error[-2]))
                self.apparent_power_error_der.append(1/self.Ts*(self.apparent_power_error[-1] - self.apparent_power_error[-2]))
                
                # PID
                self.p_set_temp = self.K_P*self.active_power_error[-1] + self.K_I*self.active_power_error_int[-1] + self.K_D*self.active_power_error_der[-1]
                
            elif self.control_mode == 'peak_shaving':

                # compute reference error - error is 0 if substation power below reference
                self.active_power_error.append(min(0,self.P_target - self.measured_active_power_lpf[-1]))
                # self.active_power_error.append(min(0,self.P_target - (self.measured_active_power_lpf[-1] - self.p_set[-1]*len(self.device_id))))
                self.reactive_power_error.append(min(0,self.Q_target - self.measured_reactive_power_lpf[-1]))
                self.apparent_power_error.append(min(0,self.S_target - self.measured_apparent_power_lpf[-1]))

                # compute integral of reference error
                # if substation power above reference, integrate error
                # else, exponential decay integral of error
                if self.active_power_error[-1] < 0:
                    self.active_power_error_int.append(self.active_power_error_int[-2] + self.Ts*self.active_power_error[-2])
                    self.reactive_power_error_int.append(self.reactive_power_error_int[-2] + self.Ts*self.reactive_power_error[-2])
                    self.apparent_power_error_int.append(self.apparent_power_error_int[-2] + self.Ts*self.apparent_power_error[-2])
                else:
                    self.active_power_error_int.append((1 - self.K_I*self.Ts)*self.active_power_error_int[-2])
                    self.reactive_power_error_int.append((1 - self.K_I*self.Ts)*self.reactive_power_error_int[-2])
                    self.apparent_power_error_int.append((1 - self.K_I*self.Ts)*self.apparent_power_error_int[-2])

                # compute derivative of reference error
                self.active_power_error_der.append(1/self.Ts*(self.active_power_error[-1] - self.active_power_error[-2]))
                self.reactive_power_error_der.append(1/self.Ts*(self.reactive_power_error[-1] - self.reactive_power_error[-2]))
                self.apparent_power_error_der.append(1/self.Ts*(self.apparent_power_error[-1] - self.apparent_power_error[-2]))

                # PID
                self.p_set_temp = self.K_P*self.active_power_error[-1] + self.K_I*self.active_power_error_int[-1] + self.K_D*self.active_power_error_der[-1]

            elif self.control_mode == 'valley_filling':

                # compute reference error - error is 0 if substation power above reference
                self.active_power_error.append(max(0,self.P_target - self.measured_active_power_lpf[-1]))
                # self.active_power_error.append(min(0,self.P_target - (self.measured_active_power_lpf[-1] - self.p_set[-1]*len(self.device_id))))
                self.reactive_power_error.append(max(0,self.Q_target - self.measured_reactive_power_lpf[-1]))
                self.apparent_power_error.append(max(0,self.S_target - self.measured_apparent_power_lpf[-1]))

                # compute integral of reference error
                # if substation power above reference, integrate error
                # else, exponential decay integral of error
                if self.active_power_error[-1] > 0:
                    self.active_power_error_int.append(self.active_power_error_int[-2] + self.Ts*self.active_power_error[-2])
                    self.reactive_power_error_int.append(self.reactive_power_error_int[-2] + self.Ts*self.reactive_power_error[-2])
                    self.apparent_power_error_int.append(self.apparent_power_error_int[-2] + self.Ts*self.apparent_power_error[-2])
                else:
                    self.active_power_error_int.append((1 - self.K_I*self.Ts)*self.active_power_error_int[-2])
                    self.reactive_power_error_int.append((1 - self.K_I*self.Ts)*self.reactive_power_error_int[-2])
                    self.apparent_power_error_int.append((1 - self.K_I*self.Ts)*self.apparent_power_error_int[-2])

                # compute derivative of reference error
                self.active_power_error_der.append(1/self.Ts*(self.active_power_error[-1] - self.active_power_error[-2]))
                self.reactive_power_error_der.append(1/self.Ts*(self.reactive_power_error[-1] - self.reactive_power_error[-2]))
                self.apparent_power_error_der.append(1/self.Ts*(self.apparent_power_error[-1] - self.apparent_power_error[-2]))

                # PID
                self.p_set_temp = self.K_P*self.active_power_error[-1] + self.K_I*self.active_power_error_int[-1] + self.K_D*self.active_power_error_der[-1]

            # setpoint
            self.p_set.append(1*self.p_set_temp)

            result = {}

            max_charge_power_list = []
            max_discharge_power_list = []

            # Obtain list of BSD max charge and discharge powers
            for device in self.device_id:

                max_charge_power_list.append(env.k.device.devices[device]['device'].max_charge_power/1e3)
                max_discharge_power_list.append(env.k.device.devices[device]['device'].max_discharge_power/1e3)

            # Iterate through BSDs
            for device in self.device_id:

                # charge
                if self.p_set[-1] >= 0:

                    # if self.p_set[-1] >= np.abs(self.measured_active_power_lpf[-2] - self.P_target):
                    #     self.p_set[-1] = np.abs(self.measured_active_power_lpf[-2] - self.P_target)

                    if self.p_set[-1] >= max(max_charge_power_list):
                        self.p_set[-1] = max(max_charge_power_list)

                    # limit charge power assigned to BSD to BSD max charge power
                    if self.p_set[-1] >= env.k.device.devices[device]['device'].max_charge_power/1e3:
                        # self.p_set[-1] = env.k.device.devices[device]['device'].max_charge_power/1e3
                        result[device] = (None, {'control_mode': 'charge', 'p_in': env.k.device.devices[device]['device'].max_charge_power/1e3})
                    else:
                        result[device] = (None, {'control_mode': 'charge', 'p_in': 1*self.p_set[-1]})
                    # if self.p_set[-1] >= (self.P_target - self.measured_active_power_lpf[-2]):
                    #     self.p_set[-1] = (self.P_target - self.measured_active_power_lpf[-2])

                    # self.control_setting.append('charge')
                    # self.p_in.append(self.p_set[-1])
                    # self.p_out.append(0)
                    # self.custom_control_setting = {'p_in': 1*self.p_in[-1]}

                # discharge
                elif self.p_set[-1] <= 0:

                    # if self.p_set[-1] <= np.abs(self.measured_active_power_lpf[-2] - self.P_target):
                    #     self.p_set[-1] = -np.abs(self.measured_active_power_lpf[-2] - self.P_target)

                    if self.p_set[-1] <= -max(max_discharge_power_list):
                        self.p_set[-1] = -max(max_discharge_power_list)

                    # limit discharge power assigned to BSD to BSD max discharge power
                    if self.p_set[-1] <= -env.k.device.devices[device]['device'].max_discharge_power/1e3:
                        # self.p_out[-1] = -env.k.device.devices[device]['device'].max_discharge_power/1e3
                        result[device] = (None, {'control_mode': 'discharge', 'p_out': env.k.device.devices[device]['device'].max_discharge_power/1e3})
                    else:
                        # self.p_out.append(self.p_set[-1])
                        result[device] = (None, {'control_mode': 'discharge', 'p_out': -1*self.p_set[-1]})

                    # if self.p_set[-1] <= (self.P_target - self.measured_active_power_lpf[-2]):
                    #     self.p_set[-1] = (self.P_target - self.measured_active_power_lpf[-2])
                    # self.control_setting.append('discharge')
                    # self.p_in.append(0)
                    # # self.p_out.append(-self.p_set[-1])
                    # self.custom_control_setting = {'p_out': 1*self.p_out[-1]}

                    # result[device] = ('discharge',  {'p_out': 1*self.p_out[-1]})

            # return result

            # if env.k.time % self.print_interval == 0:
            #     print('Time: ' + str(env.k.time))
            #     print('Controller: ' + self.controller_id)

            #     for device in self.device_id:
            #         print('Device: ' + str(device))
            #         print('Battery SOC: ' + str(env.k.device.devices[device]['device'].SOC))

            #     # print('Measured active power [kW]: ' + str(self.measured_active_power[-1]))
            #     # print('Measured reactive power [kVAr]: ' + str(self.measured_reactive_power[-1]))
            #     # print('Measured apparent power [kVA]: ' + str(self.measured_apparent_power[-1]))

            #     print('Measured active power lpf [kW]: ' + str(self.measured_active_power_lpf[-1]))
            #     print('Measured reactive power lpf [kVAr]: ' + str(self.measured_reactive_power_lpf[-1]))
            #     print('Measured apparent power lpf [kVA]: ' + str(self.measured_apparent_power_lpf[-1]))

            #     # if self.control_setting[-2] == 'charge':
            #     #     print('Charge')
            #     #     print('Active Power Control k-1 [kW]: ' + str(self.p_in[-2]))
            #     # if self.control_setting[-1] == 'discharge':
            #     #     print('Discharge')
            #     #     print('Active Power Control k-1 [kW]: ' + str(self.p_out[-2]))

            #     # print('Load active power [kW]: ' + str(self.load_active_power[-1]))
            #     # print('Load reactive power [kVAr]: ' + str(self.load_reactive_power[-1]))
            #     # print('Load apparent power [kVA]: ' + str(self.load_apparent_power[-1]))
            #     # print('')

            #     print(result)
            #     for device in self.device_id:
            #         print('Device: ' + str(device))
            #         print(result[device])

            #     # if self.control_setting[-1] == 'charge':
            #     #     print('Discharge')
            #     #     # print('Discharge power non rectified [kW]: ' + str(self.p_set[-1]))
            #     #     print('Discharge power [kW]: ' + str(self.p_in[-1]))

            #     # if self.control_setting[-1] == 'discharge':
            #     #     print('Discharge')
            #     #     # print('Discharge power non rectified [kW]: ' + str(self.p_set[-1]))
            #     #     print('Discharge power [kW]: ' + str(self.p_out[-1]))            
                

            #     print('')

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

        Logger.log(self.controller_id, 'active_power_error', self.active_power_error[-1])
        Logger.log(self.controller_id, 'active_power_error_int', self.active_power_error_int[-1])
        Logger.log(self.controller_id, 'active_power_error_der', self.active_power_error_der[-1])

        Logger.log(self.controller_id, 'p_target', self.P_target)

        Logger.log(self.controller_id, 'p_set', self.p_set[-1])
        Logger.log(self.controller_id, 'p_in', self.p_in[-1])
        Logger.log(self.controller_id, 'p_out', self.p_out[-1])