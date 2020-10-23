from pycigar.controllers.base_controller import BaseController


class CentralizedBatteryController(BaseController):
    """Fixed controller is the controller that do nothing.

    It only returns the 'default_control_setting' value when being called.

    Attributes
    ----------
    additional_params : dict
        The parameters of the controller
    """

    def __init__(self, device_id, additional_params, is_disable_log=False):
        """Instantiate an fixed Controller."""
        BaseController.__init__(
            self,
            device_id
        )
        self.additional_params = additional_params
        self.list_devices = device_id
        self.count = 0

    def get_action(self, env):
        """See parent class."""
        if self.count == 0:
            charge = True
            discharge = True
            for device_id in self.list_devices:
                if env.k.device.devices[device_id]['device'].SOC >= 0.2:
                    charge = charge & False
                elif env.k.device.devices[device_id]['device'].SOC <= 0.8:
                    discharge = discharge & False
                else:
                    control_mode = env.k.device.devices[device_id]['device'].control_mode
            if charge:
                control_mode = 'charge'
            elif discharge:
                control_mode = 'discharge'
            self.control_mode = control_mode
            
        elif self.count < len(self.list_devices)-1:
            self.count += 1
        else:
            self.count = 0

        return self.control_mode, {'pout': 10}

#         return self.additional_params['default_control_setting']

    def reset(self):
        """See parent class."""
        pass
