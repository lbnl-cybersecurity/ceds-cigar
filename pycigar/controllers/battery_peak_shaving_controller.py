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

    def get_action(self, env):
        """See parent class."""
        power_reactive_power = env.k.kernel_api.get_total_power()
        if env.k.device.devices[self.device_id]['device'].SOC <= 0.2:
#             print('CHARGE')
            control_setting = 'charge'
        elif env.k.device.devices[self.device_id]['device'].SOC >= 0.8:
#             print('DISCHARGE')
            control_setting = 'discharge'
        else:
            control_setting = env.k.device.devices[self.device_id]['device'].control_setting

        return control_setting

    def reset(self):
        """See parent class."""
        pass