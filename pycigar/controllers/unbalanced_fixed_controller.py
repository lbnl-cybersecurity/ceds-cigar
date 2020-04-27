import numpy as np
from pycigar.controllers.base_controller import BaseController


class UnbalancedFixedController(BaseController):
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
        self.trigger = False
        self.hack_curve_all = np.array([1.01, 1.04, 1.04, 1.07, 1.09])
        self.hack_curve_a = np.array([1.01, 1.04, 1.04, 1.07, 1.09]) + 0.1
        self.hack_curve_b = np.array([0.95, 0.98, 0.98, 1.01, 1.04])
        self.hack_curve_c = np.array([0.95, 0.98, 0.98, 1.01, 1.04]) - 0.1
        self.action = None

    def get_action(self, env):
        """See parent class."""
        # nothing to do here, the setting in the device is as default
        if self.trigger is False:
            if self.device_id[-1].isdigit():
                self.action = self.hack_curve_all
            elif self.device_id[-1] == 'a':
                self.action = self.hack_curve_a
            elif self.device_id[-1] == 'b':
                self.action = self.hack_curve_b
            elif self.device_id[-1] == 'c':
                self.action = self.hack_curve_c

            self.trigger = True
            return self.action
        else:
            return self.action

    def reset(self):
        """See parent class."""
        self.trigger = False
