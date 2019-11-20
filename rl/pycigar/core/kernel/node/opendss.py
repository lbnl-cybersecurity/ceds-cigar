from pycigar.core.kernel.node import KernelNode
import math
import numpy as np
import random

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
            self.nodes[node] = {
                "voltage": None,
                "load": None,
                "PQ_injection": {"P": 0, "Q": 0}
            }

    def update(self, reset):
        """See parent class."""
        pf_converted = math.tan(math.acos(0.9))
        if reset is True:
            for node in self.nodes:
                self.nodes[node]['voltage'] = np.zeros(len(self.nodes[node]['load']))
                self.nodes[node]['PQ_injection'] = {"P": 0, "Q": 0}
                self.kernel_api.set_node_kw(node, self.nodes[node]["load"][0])
                self.kernel_api.set_node_kvar(node, self.nodes[node]["load"][0]*pf_converted)
        else:
            for node in self.nodes:
                self.kernel_api.set_node_kw(node,
                                            self.nodes[node]["load"]
                                            [self.master_kernel.time] +
                                            self.nodes[node]["PQ_injection"]['P'])
                self.kernel_api.set_node_kvar(node,
                                              self.nodes[node]["load"]
                                              [self.master_kernel.time]*pf_converted +
                                              self.nodes[node]["PQ_injection"]['Q'])

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
        return self.nodes[node_id]['voltage'][self.master_kernel.time-1]

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
        return self.nodes[node_id]['load'][self.master_kernel.time-1]

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
