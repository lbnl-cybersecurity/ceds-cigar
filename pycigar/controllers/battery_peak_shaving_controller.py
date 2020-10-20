from pycigar.controllers.base_controller import BaseController


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

    def get_action(self, env):

        # print(env.k.time)
        if env.k.time % 600 == 0:
            print('Time: ' + str(env.k.time))

        total_active_power = -env.k.kernel_api.get_total_power()[0]
        total_reactive_power = -env.k.kernel_api.get_total_power()[1]

        # print(total_apparent_power)
        total_apparent_power = (total_active_power**2 + total_reactive_power**2)**(1/2)
        # print(total_apparent_power)

        if env.k.time % 600 == 0:
            print('Time: ' + str(env.k.time))
            print('Active Power: ' + str(total_active_power))
            print('Reactive Power: ' + str(total_reactive_power))
            print('Apparent Power: ' + str(total_apparent_power))
            print('')
        
        if total_apparent_power >= 8e2:
            control_setting = 'discharge'
            # pout = max_discharge - total_apparent_power
        else:
            control_setting = 'charge'
        if env.k.device.devices[self.device_id]['device'].SOC <= 0.2:
#             print('CHARGE')
            control_setting = 'charge'

        return control_setting, {'pout': total_apparent_power}


    def reset(self):
        """See parent class."""
        pass