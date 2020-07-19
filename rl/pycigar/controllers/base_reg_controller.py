from pycigar.controllers.base_controller import BaseController


class BaseRegController(BaseController):
    """Fixed controller is the controller that do nothing.

    It only returns the 'default_control_setting' value when being called.

    Attributes
    ----------
    additional_params : dict
        The parameters of the controller
    """

    def __init__(self, device_id, additional_params):
        """Instantiate an fixed Controller."""
        BaseController.__init__(self, device_id)
        self.additional_params = additional_params

    def get_action(self, env):
        """See parent class."""
        # nothing to do here, the setting in the device is as default
        return self.additional_params['default_control_setting']

    def reset(self):
        """See parent class."""
        pass
