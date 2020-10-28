from pycigar.devices.base_device import BaseDevice
from pycigar.utils.logging import logger


class RegulatorDevice(BaseDevice):
    def __init__(self, device_id, additional_params=None, is_disable_log=False):
        """Instantiate an Regulator device."""
        BaseDevice.__init__(self, device_id, additional_params)

        self.is_disable_log = is_disable_log
        self.max_tap_change = additional_params.get('max_tap_change', None)
        self.forward_band = additional_params.get('forward_band', None)
        self.tap_number = additional_params.get('tap_number', None)
        self.tap_delay = additional_params.get('tap_delay', None)
        self.delay = additional_params.get('delay', None)
        self.kernel_api = additional_params.get('kernel_api')
        self.kernel_api.set_regulator_property(
            device_id,
            {
                'max_tap_change': self.max_tap_change,
                'forward_band': self.forward_band,
                'tap_number': self.tap_number,
                'tap_delay': self.tap_delay,
                'delay': self.delay,
            },
        )

    def update(self, k):
        self.log()

    def reset(self):
        self.__init__(self.device_id, self.init_params)
        self.log()

    def set_control_setting(self, control_setting):
        """See parent class."""
        pass

    def log(self):
        """See parent class."""
        Logger = logger()
        Logger.log(self.device_id, 'max_tap_change', self.max_tap_change)
        Logger.log(self.device_id, 'forward_band', self.kernel_api.get_regulator_forwardband(self.device_id))
        Logger.log(self.device_id, 'tap_number', self.kernel_api.get_regulator_tap(self.device_id))
        Logger.log(self.device_id, 'tap_delay', self.tap_delay)
        Logger.log(self.device_id, 'regulator_forwardvreg', self.kernel_api.get_regulator_forwardvreg(self.device_id))
