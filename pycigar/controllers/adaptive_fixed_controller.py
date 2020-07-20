import numpy as np
from pycigar.controllers.base_controller import BaseController


class AdaptiveFixedController(BaseController):
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
        self.hack_curve = np.array([-0.0001, 0.000, 0.000, 0.0001, 0.0002])
        self.average_span = 10

        self.action = None

    def get_action(self, env):
        """See parent class."""
        # nothing to do here, the setting in the device is as default
        if self.trigger is False:
            node_id = env.k.device.devices[self.device_id]['node_id']
            if env.k.time - self.average_span - 1 > 0:
                vk = np.mean(np.abs(env.k.node.nodes[node_id]['voltage'][env.k.time - self.average_span - 1:env.k.time - 1]))
            else:
                vk = np.mean(np.abs(env.k.node.nodes[node_id]['voltage'][0:env.k.time - 1]))
            self.action = vk + self.hack_curve
            self.trigger = True
            return self.action
        else:
            return self.action

    def reset(self):
        """See parent class."""
        self.trigger = False
