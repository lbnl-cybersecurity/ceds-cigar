from pycigar.core.kernel.device import KernelDevice
from pycigar.devices import PVDevice
from pycigar.devices import RegulatorDevice

from pycigar.controllers import AdaptiveInverterController
from pycigar.controllers import FixedController
from pycigar.controllers import RLController
import numpy as np

class OpenDSSDevice(KernelDevice):
    """See parent class.

    Attributes
    ----------
    adaptive_device_ids : list
        List of adaptive device ids controlled by adaptive controllers
    adversary_adaptive_device_ids : list
        List of adversary adaptive device ids controlled by adaptive attackers
    adversary_fixed_device_ids : list
        List of adversary fixed device ids controlled by fixed controllers
    adversary_rl_device_ids : list
        List of adversary RL device ids controlled by RL attackers
    all_device_ids : list
        List of all device ids (only the friendly devices)
    devices : dict
        A dictionary of devices: 'device_id': {
                                              'device': device_obj,
                                              'controller': controller,
                                              'node_id': node_id
                                              }
    fixed_device_ids : list
        List of fixed device ids controlled by fixed controllers
    kernel_api : any
        an API that is used to interact with the simulator
    num_adaptive_devices : int
        Number of friendly adaptive devices in the grid
    num_adversary_adaptive_devices : int
        Number of attacking adaptive devices in the grid
    num_adversary_fixed_devices : int
        Number of attacking fixed devices in the grid
    num_adversary_rl_devices : int
        Number of attacking RL devices in the grid
    num_devices : int
        Number of all devices in the grid
    num_fixed_devices : int
        Number of fixed devices in the grid
    num_pv_devices : int
        Number of PV devices in the grid
    num_rl_devices : int
        Number of devices controlled by RL controllers in the grid
    opendss_proc : process_id
        The process id of the opendss process (Deprecated)
    pv_device_ids : list
        List of PV device ids
    rl_device_ids : list
        List of RL controlled device ids
    """

    def __init__(self, master_kernel):
        """See parent class."""
        KernelDevice.__init__(self, master_kernel)
        self.opendss_proc = None  # depricated, not in use
        self.devices = {}

        self.all_device_ids = []
        self.num_devices = 0
        self.pv_device_ids = []
        self.num_pv_devices = 0

        self.rl_device_ids = []
        self.num_rl_devices = 0
        self.adaptive_device_ids = []
        self.num_adaptive_devices = 0
        self.fixed_device_ids = []
        self.num_fixed_devices = 0

        self.adversary_rl_device_ids = []
        self.num_adversary_rl_devices = 0
        self.adversary_adaptive_device_ids = []
        self.num_adversary_adaptive_devices = 0
        self.adversary_fixed_device_ids = []
        self.num_adversary_fixed_devices = 0

        self.regulator_device_ids = []
        self.num_regulator_device_ids = 0

    def start_device(self):
        self.opendss_proc = None  # depricated, not in use
        self.devices = {}

        self.all_device_ids = []
        self.num_devices = 0
        self.pv_device_ids = []
        self.num_pv_devices = 0

        self.rl_device_ids = []
        self.num_rl_devices = 0
        self.adaptive_device_ids = []
        self.num_adaptive_devices = 0
        self.fixed_device_ids = []
        self.num_fixed_devices = 0

        self.adversary_rl_device_ids = []
        self.num_adversary_rl_devices = 0
        self.adversary_adaptive_device_ids = []
        self.num_adversary_adaptive_devices = 0
        self.adversary_fixed_device_ids = []
        self.num_adversary_fixed_devices = 0

        self.regulator_device_ids = []
        self.num_regulator_device_ids = 0
        
    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def add(self, name, connect_to, device=(PVDevice, {}), controller=(AdaptiveInverterController, {}),
            adversary_controller=None, hack=None):
        """Add a new device with controller into the grid connecting to a node.

        If not specifying the adversarial controller and hack, it implies that
        there is no hack at the node. Otherwise, we will create 2 separate
        devices controlled by adaptive controllers with the same setting as
        friendly adaptive controller and with the percentage control is devided
        into 2 parts at hacked time; we also create hacked controller but set
        to inactive and flip the 2 controllers at hack time.

        Parameters
        ----------
        name : string
            The device name, eventually we use it for device id. For an
            adversarial device, the id is 'adversary_name'
        connect_to : string
            Node id, where the device is connected to
        device : list, optional
            List of device type and its parameters
        controller : list, optional
            List of controller type and its parameters
        adversary_controller : None, optional
            List of adversarial controller type and its parameters
        hack : list, optional
            List of percentage hack and hack timestep

        Returns
        -------
        string
            Adversarial device id, ad-hoc return need to be fixed
        """
        device_id = name

        if device[0] == RegulatorDevice:
            device_obj = device[0](device_id, device[1])
            self.devices[device_id] = {"device": device_obj}
            self.regulator_device_ids.append(device_id)
            self.num_regulator_device_ids += 1
            self.all_device_ids.extend(device_id)
            self.num_devices += 1
            return None

        # create ally device
        if hack is None:
            device[1]["percentage_control"] = 1
        else:
            device[1]["percentage_control"] = 1 - hack[1]
        device_obj = device[0](device_id, device[1])
        if device[0] == PVDevice:
            self.pv_device_ids.append(device_id)
            self.num_pv_devices += 1

        controller_obj = controller[0](device_id, additional_params=controller[1])

        if controller[0] == AdaptiveInverterController:
            self.adaptive_device_ids.append(device_id)
            self.num_adaptive_devices += 1
        elif controller[0] == FixedController:
            self.fixed_device_ids.append(device_id)
            self.num_fixed_devices += 1
        elif controller[0] == RLController:
            self.rl_device_ids.append(device_id)
            self.num_rl_devices += 1

        self.devices[device_id] = {
            "device": device_obj,
            "controller": controller_obj,
            "node_id": connect_to
        }

        # create adsersarial controller

        if adversary_controller is not None:
            adversary_device_id = "adversary_%s" % name
            device[1]["percentage_control"] = hack[1]
            adversary_device_obj = device[0](adversary_device_id, device[1])

            if device[0] == PVDevice:
                self.pv_device_ids.append(adversary_device_id)
                self.num_pv_devices += 1

            adversary_controller_obj = \
                adversary_controller[0](adversary_device_id, adversary_controller[1])

            if adversary_controller[0] == AdaptiveInverterController:
                self.adversary_adaptive_device_ids.append(adversary_device_id)
                self.num_adversary_adaptive_devices += 1
            if adversary_controller[0] == FixedController:
                self.adversary_fixed_device_ids.append(adversary_device_id)
                self.num_adversary_fixed_devices += 1
            if adversary_controller[0] == RLController:
                self.adversary_rl_device_ids.append(adversary_device_id)
                self.num_adversary_rl_devices += 1

            self.devices[adversary_device_id] = {
                "device": adversary_device_obj,
                "controller": adversary_controller_obj,
                "node_id": connect_to,
                "hack_controller": AdaptiveInverterController(adversary_device_id, controller[1])
            }
        else:
            adversary_device_id = "adversary_%i" % name
            device[1]["percentage_control"] = hack[1]
            adversary_device_obj = device[0](adversary_device_id, device[1])

            if device[0] == PVDevice:
                self.pv_device_ids.append(adversary_device_id)
                self.num_pv_devices += 1

            adversary_controller_obj = FixedController(adversary_device_id, {})
            self.adversary_fixed_device_ids.append(adversary_device_id)
            self.num_adversary_fixed_devices += 1

            self.devices[adversary_device_id] = {
                "device": adversary_device_obj,
                "controller": adversary_controller_obj,
                "node_id": connect_to,
                "hack_controller": AdaptiveInverterController(adversary_device_id, controller[1])
            }

        self.all_device_ids.extend((device_id, adversary_device_id))
        self.num_devices += 1

        return adversary_device_id

    def update(self, reset):
        """See parent class."""
        if reset is True:
            # reset device and controller
            for device_id in self.devices.keys():
                if isinstance(self.devices[device_id]['device'], PVDevice):
                    
                    self.devices[device_id]['device'].reset()
                    self.devices[device_id]['controller'].reset()
                    if 'hack_controller' in self.devices[device_id]:
                        self.devices[device_id]['hack_controller'].reset()

                        temp = self.devices[device_id]['controller']
                        self.devices[device_id]['controller'] = self.devices[device_id]['hack_controller']
                        self.devices[device_id]['hack_controller'] = temp
                elif isinstance(self.devices[device_id]['device'], RegulatorDevice):
                    self.devices[device_id]['device'].reset()

            self.total_pv_device_inject = {}
            for pv_device in self.pv_device_ids:
                self.total_pv_device_inject[pv_device] = np.array([0., 0.])

        else:
            # get the injection here
            # get the new VBP, then push PV to node
            # update pv device
            for pv_device in self.pv_device_ids:
                self.devices[pv_device]["device"].update(self.master_kernel)
                self.total_pv_device_inject[pv_device] += [self.devices[pv_device]["device"].p_out[1], self.devices[pv_device]["device"].q_out[1]]

    def get_regulator_device_ids(self):
        return self.regulator_device_ids

    def get_adaptive_device_ids(self):
        """Get all adaptive device ids, for both friendly adaptive device ids and adversarial adaptive device ids.

        Returns
        -------
        List
            List of all adaptive device ids
        """
        return self.adaptive_device_ids + self.adversary_adaptive_device_ids

    def get_fixed_device_ids(self):
        """Get all adaptive device ids, for both friendly adaptive device ids and adversarial adaptive device ids.

        Returns
        -------
        List
            List of all adaptive device ids
        """
        return self.fixed_device_ids + self.adversary_fixed_device_ids

    def get_pv_device_ids(self):
        """Return the list  of PV device ids.

        Returns
        -------
        list
            List of PV device ids
        """
        return self.pv_device_ids

    def get_rl_device_ids(self):
        """Return the list  of PV device ids controlled by RL agents.

        Returns
        -------
        list
            List of RL device ids
        """
        return self.rl_device_ids

    def get_solar_generation(self, device_id):
        """Return the solar generation value at the current timestep.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        float
            The solar generation value
        """
        device = self.devices[device_id]['device']
        return device.solar_generation[self.master_kernel.time-1]

    def get_node_connected_to(self, device_id):
        """Return the node id that the device connects to.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        string
            The node id
        """
        return self.devices[device_id]['node_id']

    def get_device_p_set_relative(self, device_id):
        """Return the device's power set relative to Sbar at the current timestep.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        float
            The relative power set
        """
        return self.devices[device_id]['device'].p_set[1]/self.devices[device_id]['device'].Sbar
    
    def get_device_p_injection(self, device_id):
        """Return the device's power injection at the current timestep.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        float
            The power value
        """
        return self.devices[device_id]['device'].p_out[1]

    def get_device_q_set(self, device_id):
        """Return the device's reactive power injection at the current timestep.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        float
            The reactive power value
        """
        return self.devices[device_id]['device'].q_set[1]

    def get_device_q_injection(self, device_id):
        """Return the device's reactive power injection at the current timestep.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        float
            The reactive power value
        """
        return self.devices[device_id]['device'].q_out[1]

    def get_device_y(self, device_id):
        """Return the device's y value at the current timestep.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        float
            The y value
        """
        return self.devices[device_id]['device'].y

    def get_device(self, device_id):
        """Return device object given the device id.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        pycigar.controllers.BaseDevice
            A device object
        """
        return self.devices[device_id]['device']

    def get_controller(self, device_id):
        """Return the controller given the device id.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        pycigar.controllers.BaseController
            A controller object
        """
        return self.devices[device_id]['controller']

    def get_control_setting(self, device_id):
        """Return the control setting of the device.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        list
            A device's control setting, for example:
            [0.95, 1.01, 1.01, 1.01, 1.05]
        """
        return self.devices[device_id]['device'].control_setting

    def apply_control(self, device_id, control_setting):
        """Apply the control setting on a device given the device id.

        Parameters
        ----------
        device_id : string
            The device id
        control_setting : list
            The control setting of the device (e.g. VBP for PVDevice...)
        """
        if type(device_id) == str:
            device_id = [device_id]
            control_setting = [control_setting]

        for i, device_id in enumerate(device_id):
            if control_setting[i] is not None:
                device = self.devices[device_id]['device']
                device.set_control_setting(control_setting[i])

    def set_device_internal_scenario(self, device_id, internal_scenario):
        """Set device internal scenario.

        For example, PV site receives
        solar generation list as its internal scenario.

        Parameters
        ----------
        device_id : string
            The device id
        internal_scenario : list
            The device internal scenario
        """
        device = self.get_device(device_id)
        if isinstance(device, PVDevice):
            device.solar_generation = internal_scenario
