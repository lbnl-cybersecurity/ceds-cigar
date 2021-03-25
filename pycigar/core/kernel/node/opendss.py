import math

import numpy as np
from pycigar.core.kernel.node import KernelNode
from pycigar.utils.logging import logger

PF_CONVERTED = math.tan(math.acos(0.9))

class OpenDSSNode(KernelNode):
    """See parent class."""

    def __init__(self, master_kernel):
        """See parent class."""
        KernelNode.__init__(self, master_kernel)
        self.nodes = {}

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def start_nodes(self):
        """Create the dictionary of nodes to track the node information."""
        node_ids = self.kernel_api.get_node_ids()
        for node in node_ids:
            self.nodes[node] = {"voltage": None, "load": None, "PQ_injection": {"P": 0, "Q": 0}}

    def update(self, reset):
        """See parent class."""

        if reset is True:
            for node in self.nodes:
                self.nodes[node]['voltage'] = np.zeros(len(self.nodes[node]['load']))
                self.nodes[node]['PQ_injection'] = {"P": 0, "Q": 0}
                self.kernel_api.set_node_kw(node, self.nodes[node]["load"][0])
                self.kernel_api.set_node_kvar(node, self.nodes[node]["load"][0] * PF_CONVERTED)
                # add dummy node for PV
                self.kernel_api.simulation_command('New Load.{}_pv like={} Bus1={} phase={} vminpu=0.8 vmaxpu=1.2 mode=1'.format(node, node, self.kernel_api.load_to_bus1[node], self.kernel_api.load_to_numphases[node]))
                self.log(
                    node,
                    self.nodes[node]['PQ_injection']['P'],
                    self.nodes[node]['PQ_injection']['Q'],
                    self.nodes[node]["load"][0],
                    self.nodes[node]["load"][0] * PF_CONVERTED,
                )
        else:
            for node in self.nodes:
                self.kernel_api.set_node_kw(node, self.nodes[node]["load"][self.master_kernel.time])
                self.kernel_api.set_node_kw('{}_pv'.format(node), self.nodes[node]["PQ_injection"]['P'])
                self.kernel_api.set_node_kvar(node, self.nodes[node]["load"][self.master_kernel.time] * PF_CONVERTED)
                self.kernel_api.set_node_kvar('{}_pv'.format(node), self.nodes[node]["PQ_injection"]['Q'])

                self.log(
                    node,
                    self.nodes[node]['PQ_injection']['P'],
                    self.nodes[node]['PQ_injection']['Q'],
                    self.nodes[node]["load"][self.master_kernel.time] + self.nodes[node]["PQ_injection"]['P'],
                    self.nodes[node]["load"][self.master_kernel.time] * PF_CONVERTED
                    + self.nodes[node]["PQ_injection"]['Q'],
                )

    def log(self, node, p_injection, q_injection, node_kw, node_kvar):
        if not self.master_kernel.sim_params['is_disable_log']:
            Logger = logger()
            Logger.log(node, 'p', p_injection)
            Logger.log(node, 'q', q_injection)
            Logger.log(node, 'kw', node_kw)
            Logger.log(node, 'kvar', node_kvar)

    def get_node_ids(self):
        """Return all nodes' ids.

        Returns
        -------
        list
            List of node id
        """
        return list(self.nodes.keys())

    def get_node_voltage(self, node_id):
        """Return current voltage at node.

        Parameters
        ----------
        node_id : string
            Node id

        Returns
        -------
        float
            Voltage value at node at current timestep
        """
        return self.nodes[node_id]['voltage'][self.master_kernel.time - 1]

    def get_node_load(self, node_id):
        """Return current load at node.

        Parameters
        ----------
        node_id : string
            Node id

        Returns
        -------
        float
            Load value at node at current timestep
        """
        return self.nodes[node_id]['load'][self.master_kernel.time - 1]

    def set_node_load(self, node_id, load):
        """Set the load scenario at node.

        Parameters
        ----------
        node_id : string
            Node id
        load : list
            A list of load at the node at each timestep
        """
        self.nodes[node_id]['load'] = load
        self.nodes[node_id]['voltage'] = np.zeros(len(load))

    def get_node_p_injection(self, node_id):
        """Return the total power injection at the node at the current timestep.

        Parameters
        ----------
        node_id : string
            Node id

        Returns
        -------
        float
            The total power injection at the node at current timestep
        """
        return self.nodes[node_id]['PQ_injection']['P']

    def get_node_q_injection(self, node_id):
        """Return the total reactive power injection at the node at the current timestep.

        Parameters
        ----------
        node_id : string
            Node id

        Returns
        -------
        float
            The total reactive power injection at the node at current timestep
        """
        return self.nodes[node_id]['PQ_injection']['Q']

    def get_all_nodes_voltage(self):
        node_ids = list(self.nodes.keys())
        voltages = []
        for node_id in node_ids:
            voltage = self.get_node_voltage(node_id)
            voltages.append(voltage)

        return voltages
