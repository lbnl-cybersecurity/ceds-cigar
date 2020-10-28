class BaseController:
    """Base controller.

    Attributes
    ----------
    device_id : string
        The device id which the controller controls
    """

    def __init__(self, device_id):
        """Instantiate the base controller.

        Parameters
        ----------
        device_id : string
            The device id which the controller controls
        """
        if isinstance(device_id, list) and len(device_id) == 1:
            self.device_id = device_id[0]
        else:
            self.device_id = device_id

    def get_action(self, env):
        """Get the control setting that the controller may want to set for the device.

        Parameters
        ----------
        env : pycigar.envs.base.py
            The environment
        """
        raise NotImplementedError

    def reset(self):
        """Reset the controller for new episode."""
        raise NotImplementedError
