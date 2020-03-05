import numpy as np
from pycigar.devices.base_device import BaseDevice

DEFAULT_CONTROL_SETTING = np.array([0.98, 1.01, 1.02, 1.05, 1.07])


class RegulatorDevice(BaseDevice):
    def __init__(self, device_id, additional_params=None):
        """Instantiate an PV device."""
        BaseDevice.__init__(self, device_id, additional_params)


        self.max_tap_change = additional_params.get('max_tap_change', 30)
        self.forward_band = additional_params.get('forward_band', 16)
        self.tap_number = additional_params.get('tap_number', 2)
        self.tap_delay = additional_params.get('tap_delay', 0)

        self.kernel_api = additional_params.get('kernel_api')
        self.kernel_api.set_regulator_property(device_id, {'max_tap_change': self.max_tap_change,
                                                           'forward_band': self.forward_band,
                                                           'tap_number': self.tap_number,
                                                           'tap_delay': self.tap_delay})

    def update(self, k):
        pass

    def reset(self):
        self.__init__(self.device_id, self.init_params)

    def set_control_setting(self, control_setting):
        """See parent class."""
        self.control_setting = control_setting
