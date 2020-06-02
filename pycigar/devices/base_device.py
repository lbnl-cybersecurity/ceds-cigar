import copy


class BaseDevice:
    """Base device.

    Attributes
    ----------
    control_setting : list
        The control setting that is currently set on the device.
    device_id : string
        The device id of this device.
    init_params : dict
        The initial parameters of this device.
    percentage_control : float
        This has range from 0 to 1, indicate how much control at a node
        of this device.
    """

    def __init__(self, device_id, additional_params):
        """Instantiate the base device.

        Parameters
        ----------
        device_id : string
            The device id of this device.
        additional_params : dict
            The initial parameters of this device.
        """
        self.device_id = device_id

        if "percentage_control" in additional_params:
            self.percentage_control = additional_params["percentage_control"]
        else:
            self.percentage_control = 1
            additional_params["percentage_control"] = self.percentage_control

        self.init_params = copy.deepcopy(additional_params)

    def set_control_setting(self, control_setting):
        """Set the control setting of the device to a new value.

        Parameters
        ----------
        control_setting : list
            New control setting for the device.
        """
        self.control_setting = control_setting

    def update(self, kernel):
        """Update the device.

        This will run the internal algorithm of device and declare how much
        of active power and reactive power need to push into the node.
        This will update the active power and reactive power to the node
        kernel.

        Parameters
        ----------
        kernel : pycigar.core.kernel.Kernel
            The highest level of kernel to interact with node kernel module.
        """
        raise NotImplementedError

    def reset(self):
        """Reset the device.
        """
        raise NotImplementedError

    def log(self):
        """Log information at device level.
        This function need to be called at the end of __init__() and update() to log information.
        """
        pass
