from pycigar.controllers.base_controller import BaseController


class RLController(BaseController):
    """A placeholder for an RL controller.

    Nothing to do with this class, the control setting will be given by the
    RLlib when running the experiment.
    """

    def __init__(self, device_id, controller_id=None, additional_params=None):
        """Instantiate an RL Controller."""
        BaseController.__init__(self, device_id, controller_id)

    def get_action(self, env):
        """See parent class."""
        pass

    def reset(self):
        """See parent class."""
        pass
