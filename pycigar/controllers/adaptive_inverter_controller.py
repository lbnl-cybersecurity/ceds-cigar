import numpy as np
from pycigar.controllers.base_controller import BaseController
import copy


class AdaptiveInverterController(BaseController):
    """The adaptive controller for inverter.

    Attributes
    ----------
    delay_timer : int
        Delay to change the control setting in delay_timer timestep
    delta_t : int
        Ask Ciaran for a proper definition #TODO
    epsilon : float
        Ask Ciaran for a proper definition #TODO
    gain : float
        Ask Ciaran for a proper definition #TODO
    high_pass_filter : int
        Ask Ciaran for a proper definition #TODO
    init_params : dict
        Dictionary of the initial parameters of the controller
    low_pass_filter : float
        Ask Ciaran for a proper definition #TODO
    adaptive_gain : float
        Ask Ciaran for a proper definition #TODO
    psi : float
        Ask Ciaran for a proper definition #TODO
    threshold : float
        Ask Ciaran for a proper definition #TODO
    time_counter : int
        The internal counter of the controller. This is for the delay control
    up : float
        Ask Ciaran for a proper definition #TODO
    uq : float
        Ask Ciaran for a proper definition #TODO
    y : float
        Ask Ciaran for a proper definition #TODO
    """

    def __init__(self, device_id, additional_params):
        """Instantiate an adaptive inverter controller."""
        BaseController.__init__(
            self,
            device_id
        )

        self.time_counter = 0
        self.init_params = copy.deepcopy(additional_params)

        self.delay_timer = additional_params.get('delay_timer', 0)
        self.threshold = additional_params.get('threshold', 0.25)
        self.adaptive_gain = additional_params.get('adaptive_gain', 20)
        self.delta_t = additional_params.get('delta_t', 1)
        self.low_pass_filter = additional_params.get('low_pass_filter', 0.1)
        self.high_pass_filter = additional_params.get('high_pass_filter', 1)
        self.gain = additional_params.get('gain', 1e5)

        # internal observer of the controller
        self.up = np.zeros(2)
        self.uq = np.zeros(2)
        self.psi = np.zeros(self.delay_timer + 1) if self.delay_timer != 0 else np.zeros(1)
        self.epsilon = np.zeros(self.delay_timer + 1) if self.delay_timer != 0 else np.zeros(1)
        self.y = np.zeros(self.delay_timer + 1) if self.delay_timer != 0 else np.zeros(1)

    def get_action(self, env):
        """See parent class."""
        if env.k.time > 1:

            # observer
            node_id = env.k.device.devices[self.device_id]['node_id']
            vk = np.abs(env.k.node.nodes[node_id]['voltage'][env.k.time - 1])
            vkm1 = np.abs(env.k.node.nodes[node_id]['voltage'][env.k.time - 2])
            psikm1 = self.psi[self.time_counter - 1]
            epsilonkm1 = self.epsilon[self.time_counter - 1]
            ykm1 = self.y[self.time_counter - 1]
            self.psi[self.time_counter] = psik = (vk - vkm1 - (self.high_pass_filter * self.delta_t / 2 - 1) *
                                                  psikm1) / (1 + self.high_pass_filter * self.delta_t / 2)
            self.epsilon[self.time_counter] = epsilonk = self.gain * (psik ** 2)
            self.y[self.time_counter] = yk = (self.delta_t * self.low_pass_filter * (epsilonk + epsilonkm1) -
                                              (self.delta_t * self.low_pass_filter - 2) * ykm1) / \
                                             (2 + self.delta_t * self.low_pass_filter)

            if (self.delay_timer != 0 and self.time_counter + 1 == self.delay_timer) or self.delay_timer == 0:
                yk = self.y[self.time_counter]
                vk = self.psi[self.time_counter]
                vkmdelay = self.psi[0]
                self.up[1] = self.adaptive_control(self.adaptive_gain, vk, vkmdelay, self.up[0], self.threshold, yk)
                self.uq[1] = self.adaptive_control(self.adaptive_gain, vk, vkmdelay, self.uq[0], self.threshold, yk)
                self.up[0] = self.up[1]
                self.uq[0] = self.uq[1]

                old_action = self.init_params['default_control_setting']
                new_action = np.array([old_action[0] - self.uq[1],
                                       old_action[1] - self.uq[1],
                                       old_action[2] - self.uq[1],
                                       old_action[3] - self.up[1],
                                       old_action[4] - self.up[1]
                                       ])
                # reset the timer
                self.time_counter = 0
                return new_action
            else:
                self.time_counter += 1

        return None

    def adaptive_control(self, adaptive_gain, vk, vkmdelay, ukmdelay, thresh, yk):
        """Control logic of adaptive control algorithm.

        Parameters
        ----------
        adaptive_gain : TYPE
            Description
        vk : TYPE
            Description
        vkmdelay : TYPE
            Description
        ukmdelay : TYPE
            Description
        thresh : TYPE
            Description
        yk : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        delay = self.delay_timer
        if yk > thresh:
            uk = delay / 2 * adaptive_gain * (vk ** 2 + vkmdelay ** 2) + ukmdelay
        else:
            uk = ukmdelay
        return uk

    def reset(self):
        """See parent class."""
        self.__init__(self.device_id, self.init_params)
