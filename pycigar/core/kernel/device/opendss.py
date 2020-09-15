from pycigar.core.kernel.device import KernelDevice
from pycigar.devices import PVDevice
from pycigar.devices import RegulatorDevice
from pycigar.devices.vectorized_pv_inverter_device import VectorizedPVDevice
from pycigar.utils.pycigar_registration import pycigar_make
from pycigar.devices import PVDevice

from pycigar.controllers import AdaptiveInverterController
from pycigar.controllers import FixedController

# from pycigar.controllers import MimicController
from pycigar.controllers import AdaptiveFixedController
from pycigar.controllers import UnbalancedFixedController

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
        self.start_device()

    def start_device(self):
        self.opendss_proc = None  # depricated, not in use
        self.devices = {}

        self.all_device_ids = []
        self.pv_device_ids = []

        self.rl_device_ids = []
        self.adaptive_device_ids = []
        self.fixed_device_ids = []

        self.regulator_device_ids = []

        self.device_ids = {}   #device id by type
        self.devcon_ids = {
            'rl_devcon': [],
            'norl_devcon': []
        }

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def add(self, name, connect_to, device=('pv_device', None), controller=(None, None),
            adversary_controller=(None, None), hack=None):
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
            self.all_device_ids.extend(device_id)

            device_name = 'regulator_device'
            if device_name not in self.device_ids:
                self.device_ids[device_name] = [device_id]
            else:
                self.device_ids[device_name].append(device_id)
            return None

        # create ally device
        if hack is None:
            device[1]["percentage_control"] = 1
        else:
            device[1]["percentage_control"] = 1 - hack[1]

        device_obj = pycigar_make(device[0], device_id=device_id, additional_params=device[1], is_disable_log=self.master_kernel.sim_params['is_disable_log'])

        if isinstance(device_obj, PVDevice):
            if 'pv_device' not in self.device_ids:
                self.device_ids['pv_device'] = []
            if device_id not in self.device_ids['pv_device']:
                self.device_ids['pv_device'].append(device_id)
        else:
            if device[0] not in self.device_ids:
                self.device_ids[device[0]] = [device_id]
            else:
                self.device_ids[device[0]].append(device_id)

        controller_obj = pycigar_make(controller[0], device_id=device_id, additional_params=controller[1])
        if controller[0] == 'rl_controller':
            self.devcon_ids['rl_devcon'].append(device_id)
        else:
            self.devcon_ids['norl_devcon'].append(device_id)


        self.devices[device_id] = {"device": device_obj, "controller": controller_obj, "node_id": connect_to}

        # create adversarial controller

        if adversary_controller[0] is not None:
            adversary_device_id = "adversary_%s" % name
            device[1]["percentage_control"] = hack[1]
            adversary_device_obj = pycigar_make(device[0], device_id=adversary_device_id, additional_params=device[1], is_disable_log=self.master_kernel.sim_params['is_disable_log'])

            if isinstance(adversary_device_obj, PVDevice):
                if 'pv_device' not in self.device_ids:
                    self.device_ids['pv_device'] = []
                if adversary_device_id not in self.device_ids['pv_device']:
                    self.device_ids['pv_device'].append(adversary_device_id)
            else:
                if device[0] not in self.device_ids:
                    self.device_ids[device[0]] = [adversary_device_id]
                else:
                    self.device_ids[device[0]].append(adversary_device_id)

            adversary_controller_obj = pycigar_make(adversary_controller[0], device_id=adversary_device_id, additional_params=adversary_controller[1])

            if controller[0] == 'rl_controller':
                self.devcon_ids['rl_devcon'].append(adversary_device_id)
            else:
                self.devcon_ids['norl_devcon'].append(adversary_device_id)


            self.devices[adversary_device_id] = {
                "device": adversary_device_obj,
                "controller": adversary_controller_obj,
                "node_id": connect_to,
                "hack_controller": pycigar_make('fixed_controller', device_id=adversary_device_id, additional_params=controller[1])  # MimicController(adversary_device_id, device_id)    #AdaptiveInverterController(adversary_device_id, controller[1])
            }
        else:
            adversary_device_id = None

        return adversary_device_id

    def update(self, reset):
        """See parent class."""
        if reset is True:
            # reset device and controller
            for device_id in self.devices.keys():
                self.devices[device_id]['device'].reset()
                if 'controller' in self.devices[device_id]:
                    self.devices[device_id]['controller'].reset()
                if 'hack_controller' in self.devices[device_id]:
                    self.devices[device_id]['hack_controller'].reset()
                    temp = self.devices[device_id]['controller']
                    self.devices[device_id]['controller'] = self.devices[device_id]['hack_controller']
                    self.devices[device_id]['hack_controller'] = temp

                    self.update_kernel_device_info(device_id)

            if self.master_kernel.sim_params['vectorized_mode']:
                self.vectorized_pv_inverter_device = VectorizedPVDevice(self.master_kernel, is_disable_log=self.master_kernel.sim_params['is_disable_log'])
        else:
            # get the injection here
            # get the new VBP, then push PV to node
            # update pv device
            if self.master_kernel.sim_params['vectorized_mode']:
                self.vectorized_pv_inverter_device.update(self.master_kernel)
                for device_type in self.device_ids:
                    if device_type != 'pv_device':
                        for device in self.device_ids[device_type]:
                            self.devices[device]['device'].update(self.master_kernel)

            else:
                for device_type in self.device_ids:
                    for device in self.device_ids[device_type]:
                        self.devices[device]['device'].update(self.master_kernel)

    def update_kernel_device_info(self, device_id):
        if isinstance(self.devices[device_id]["controller"], RLController):
            if device_id not in self.devcon_ids['rl_devcon']:
                self.devcon_ids['rl_devcon'].append(device_id)
            if device_id in self.devcon_ids['norl_devcon']:
                self.devcon_ids['norl_devcon'].remove(device_id)
        else:
            if device_id not in self.devcon_ids['norl_devcon']:
                self.devcon_ids['norl_devcon'].append(device_id)
            if device_id in self.devcon_ids['rl_devcon']:
                self.devcon_ids['rl_devcon'].remove(device_id)

    def get_pv_device_ids(self):
        """Return the list  of PV device ids controlled by RL agents.

        Returns
        -------
        list
            List of RL device ids
        """
        if 'pv_device' in self.device_ids:
            return self.device_ids['pv_device']
        else:
            return []

    def get_regulator_device_ids(self):
        """Return the list  of PV device ids controlled by RL agents.

        Returns
        -------
        list
            List of RL device ids
        """
        if 'pv_device' in self.device_ids:
            return self.device_ids['regulator_device']
        else:
            return []

    def get_rl_device_ids(self):
        """Return the list  of PV device ids controlled by RL agents.

        Returns
        -------
        list
            List of RL device ids
        """
        return self.devcon_ids['rl_devcon']

    def get_norl_device_ids(self):
        """Return the list  of PV device ids controlled by RL agents.

        Returns
        -------
        list
            List of RL device ids
        """
        return self.devcon_ids['norl_devcon']

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
        return device.solar_generation[self.master_kernel.time - 1]

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
        return self.devices[device_id]['device'].p_set[1] / self.devices[device_id]['device'].Sbar

    def get_device_p_set_p_max(self, device_id):
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
        return self.devices[device_id]['device'].p_set[1] / max(10, self.devices[device_id]['device'].solar_irr)

    def get_device_sbar_solar_irr(self, device_id):
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
        return (
            abs(self.devices[device_id]['device'].Sbar ** 2 - max(10, self.devices[device_id]['device'].solar_irr) ** 2)
        ) ** (1 / 2)

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

    def get_device_u(self, device_id):
        """Return the device's u value at the current timestep.

        Parameters
        ----------
        device_id : string
            The device id

        Returns
        -------
        float
            The u value
        """
        return self.devices[device_id]['device'].u

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
                device.set_control_setting(control_setting[i]) if type(control_setting[i]) is not tuple else device.set_control_setting(*control_setting[i])

                if self.master_kernel.sim_params['vectorized_mode']:
                    self.vectorized_pv_inverter_device.set_control_setting(device_id, control_setting[i])


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

    def set_device_sbar(self, device_id, sbar):
        device = self.get_device(device_id)
        if isinstance(device, PVDevice):
            device.Sbar = 1.1*sbar