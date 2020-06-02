from pycigar.controllers.base_controller import BaseController


class BatteryStorageController01(BaseController):
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
        
        if env.k.device.devices[self.device_id]['device'].SOC <= 0.2:
            control_mode = 'charge'
        if env.k.device.devices[self.device_id]['device'].SOC >= 0.8:
            control_mode = 'discharge'
        
        return control_mode
        
#         return self.additional_params['default_control_setting']

    def reset(self):
        """See parent class."""
        pass
