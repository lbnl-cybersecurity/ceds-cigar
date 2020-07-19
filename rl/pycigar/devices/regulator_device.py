from pycigar.devices.base_device import BaseDevice


class RegulatorDevice(BaseDevice):
    def __init__(self, device_id, additional_params=None, _kernel=None):
        """Instantiate an PV device."""
        BaseDevice.__init__(self, device_id, additional_params)

        self.control_setting = None
        # print('Additional Params for Regulators:', additional_params)
        if "name" in additional_params:
            self.name = additional_params["name"]
        else:
            print('While defining Regulators you have to provide a valid name')
            raise SystemError

        self.tap_delay = additional_params["tap_delay"] if "tap_delay" in additional_params else 30
        _kernel.kernel_api.set_regulator_property(device_id, {"tap_delay": self.tap_delay})

        self.tap_number = additional_params["tap_number"] if "tap_number" in additional_params else 0
        _kernel.kernel_api.set_regulator_property(device_id, {"tap_number": self.tap_number})

        self.forward_band = additional_params["forward_band"] if "forward_band" in additional_params else 0.2
        _kernel.kernel_api.set_regulator_property(device_id, {"forward_band": self.forward_band})

        self.max_tap_change = additional_params["max_tap_change"] if "max_tap_change" in additional_params else 2
        _kernel.kernel_api.set_regulator_property(device_id, {"max_tap_change": self.max_tap_change})

    def update(self, kernel):
        """See parent class."""
        all_regs = kernel.sim_params['scenario_config']['regs']
        for regs in all_regs:
            if regs['name'] == self.device_id:
                device = regs['devices'][0]
                if kernel.time == device['hack'][0]:
                    #print (kernel.time)
                    #print(type(device['adversary_custom_configs']))
                    for key, value in device['adversary_custom_configs'].items():
                        kernel.kernel_api.set_regulator_property(self.device_id, {key: value})
                        print('Updating Device Property at {} for reg {} with property {}'.format(kernel.time, key, value))
        # if kernel.time == prop0
        #
        # pass

    def reset(self):
        """See parent class."""

        additional_params = self.init_params

        if "tap_delay" in additional_params:
            self.tap_delay = additional_params["tap_delay"]
        else:
            self.tap_delay = 30

        if "tap_number" in additional_params:
            self.tap_number = additional_params["tap_number"]
        else:
            self.tap_number = 0

        if "forward_band" in additional_params:
            self.forward_band = additional_params["forward_band"]
        else:
            self.forward_band = 0.2

        if "max_tap_change" in additional_params:
            self.max_tap_change = additional_params["max_tap_change"]
        else:
            self.max_tap_change = 2

    def set_control_setting(self, control_setting):
        """See parent class."""
        self.control_setting = control_setting
