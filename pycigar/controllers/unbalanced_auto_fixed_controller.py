import numpy as np
from pycigar.controllers.base_controller import BaseController


class UnbalancedAutoFixedController(BaseController):
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
        self.trigger = False
        self.hack_curve_all = np.array([0.98, 1.01, 1.01, 1.04, 1.07])
        self.hack_curve_a = np.array([0.98, 1.01, 1.01, 1.04, 1.07]) - 0.1
        self.hack_curve_b = np.array([0.98, 1.01, 1.01, 1.04, 1.07]) + 0.1
        self.hack_curve_c = np.array([0.98, 1.01, 1.01, 1.04, 1.07]) - 0.1
        self.action = None
        self.countdown_timer = 0
        self.countdown = 40

    def get_action(self, env):
        """See parent class."""
        # nothing to do here, the setting in the device is as default
        #env.k.device.vectorized_pv_inverter_device.y[index] < 0.02 and
        init_action = env.INIT_ACTION[self.device_id]
        bus = env.k.kernel_api.load_to_bus[self.device_id.split('_')[-1]]
        if env.k.kernel_api.u_bus[bus] < 0.012 and self.countdown_timer >= self.countdown:
            self.trigger = False
            self.countdown_timer = 0

        self.countdown_timer += 1

        if self.trigger is False:
            if self.device_id[-1].isdigit():
                self.action = self.hack_curve_all
            elif self.device_id[-1] == 'a':
                if (self.action == init_action).all() or self.action is None:
                    self.action = self.hack_curve_a
                else:
                    self.action = init_action
            elif self.device_id[-1] == 'b':
                if (self.action == init_action).all() or self.action is None:
                    self.action = self.hack_curve_b
                else:
                    self.action = init_action
            elif self.device_id[-1] == 'c':
                if (self.action == init_action).all() or self.action is None:
                    self.action = self.hack_curve_c
                else:
                    self.action = init_action

            self.trigger = True
            return self.action
        else:
            return self.action

    def reset(self):
        """See parent class."""
        self.trigger = False
