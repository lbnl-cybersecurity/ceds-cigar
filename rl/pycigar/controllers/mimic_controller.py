from pycigar.controllers.base_controller import BaseController


class MimicController(BaseController):
    """Fixed controller is the controller that do nothing.

    It only returns the 'default_control_setting' value when being called.

    Attributes
    ----------
    additional_params : dict
        The parameters of the controller
    """

    def __init__(self, device_id, host_device_id):
        """Instantiate an fixed Controller."""
        BaseController.__init__(
            self,
            device_id
            )
        self.host_device_id = host_device_id

    def get_action(self, env):
        """See parent class."""
        # nothing to do here, the setting in the device is as default
        return env.k.device.get_control_setting(self.host_device_id)

    def reset(self):
        """See parent class."""
        pass
