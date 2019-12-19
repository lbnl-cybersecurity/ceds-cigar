from pycigar.core.kernel.simulation import OpenDSSSimulation
from pycigar.core.kernel.scenario import OpenDSSScenario
from pycigar.core.kernel.node import OpenDSSNode
from pycigar.core.kernel.device import OpenDSSDevice

from pycigar.utils.exeptions import FatalPyCIGARError
import numpy as np


class Kernel(object):
    """Kernel for abstract function calling across grid simulator APIs.

    The kernel contains four different subclasses for distinguishing between
    the various components of a traffic simulator.
    * simulation: controls starting, loading, saving, advancing, and resetting
      the simulator (see pycigar/core/kernel/simulation/base.py)
    * scenario: generates components for an experiment. (see
      pycigar/core/kernel/scenario/base.py)
    * device: stores, regularly updates device information, apply control
      on devices. (see pycigar/core/kernel/vehicle/base.py).
    * node: stores and regularly updates node information
      (see pycigar/core/kernel/traffic_light/base.py).

    The above kernel subclasses are designed specifically to support
    simulator-agnostic state information calling. For example, if you would
    like to collect the control setting of a specific device, then simply type:
    >>> k = Kernel(simulator="...")  # a kernel for some simulator type
    >>> device_id = "..."  # some vehicle ID
    >>> control_setting = k.device.get_control_setting(device_id)
    In addition, these subclasses support sending commands to the simulator via
    its API. For example, in order to assign a specific vehicle a target
    acceleration, type:
    >>> k = Kernel(simulator="...")  # a kernel for some simulator type
    >>> device_id = "..."  # some vehicle ID
    >>> control_setting = "..."  # some vehicle ID
    >>> k.device.apply_cotrol(device_id, control_setting)

    These subclasses can be modified and recycled to support various different
    grid simulators, e.g. OpenDSS, Gridlab-D & OMF...
    """

    def __init__(self, simulator, sim_params):
        """Instantiate a PyCIGAR kernel object.

        Parameters
        ----------
        simulator : string
            The name of simulator we would like to use. For now, there is only
            "opendss".
        sim_params : dict
            The simulation parameters of the experiment.

        Raises
        ------
        FatalPyCIGARError
            The simulator is unkown.
        """
        self.kernel_api = None
        self.sim_params = sim_params
        self.time = 0

        if simulator == "opendss":
            self.simulation = OpenDSSSimulation(self)
            self.scenario = OpenDSSScenario(self)
            self.node = OpenDSSNode(self)
            self.device = OpenDSSDevice(self)
        else:
            raise FatalPyCIGARError("Simulator type '{}' is not valid.".format(simulator))

    def pass_api(self, kernel_api):
        """Pass the API to kernel subclasses."""
        self.kernel_api = kernel_api
        self.simulation.pass_api(kernel_api)
        self.scenario.pass_api(kernel_api)
        self.node.pass_api(kernel_api)
        self.device.pass_api(kernel_api)

    def update(self, reset):
        """Call update for each simulator step.

        Parameters
        ----------
        reset : bool
            specifies whether the simulator was reset in the last simulation
            step.
        """
        if reset is True:
            start_time, end_time = self.sim_params['scenario_config']['start_end_time']
            self.t = end_time - start_time
            self.time = 0
            self.device.update(reset)
            self.scenario.change_load_profile(start_time, end_time)
            self.node.update(reset)
            self.simulation.update(reset)
            self.scenario.update(reset)
            self.warm_up()

        else:
            self.device.update(reset)  # calculate new PQ with new VBP, then push PV to node
            self.node.update(reset)  # with the load, update the load-pq to simulator
            self.simulation.update(reset)  # run a simulation step
            self.scenario.update(reset)  # update voltage on node
        self.time += 1

    def close(self):
        """Close the simulation and simulator."""
        self.simulation.close()

    def warm_up(self):
        """Run the simulation until the voltage is stablized."""
        voltages = self.node.get_all_nodes_voltage()
        self.time += 1
        self.device.update(reset=False)
        self.node.update(reset=False)
        self.simulation.update(reset=False)
        self.scenario.update(reset=False)
        while any(abs(deltaV) > 1e-5 for deltaV in np.array(self.node.get_all_nodes_voltage()) - np.array(voltages)):
            voltages = self.node.get_all_nodes_voltage()
            self.time += 1
            self.device.update(reset=False)
            self.node.update(reset=False)
            self.simulation.update(reset=False)
            self.scenario.update(reset=False)
