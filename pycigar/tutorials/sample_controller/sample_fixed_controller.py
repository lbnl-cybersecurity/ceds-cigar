import numpy as np
from pycigar.controllers.base_controller import BaseController


class SampleFixedController(BaseController):
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

    def get_action(self, env=None):
        """See parent class."""
        # nothing to do here, the setting in the device is as default
        return np.array(self.additional_params['default_control_setting']) + 0.1

    def reset(self):
        """See parent class."""
        pass
