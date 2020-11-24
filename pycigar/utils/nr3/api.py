"""Contains the Power/opendss API manager."""
from nr3 import NR3
import numpy as np
import warnings
from pycigar.utils.logging import logger
import opendssdirect as dss

class PyCIGARNR3API(object):
    """An API used to interact with OpenDSS via a TCP connection."""

    def __init__(self, port):
        """Instantiate the API.

        Parameters
        ----------
        port : int
            the port number of the socket connection
        """
        self.port = port

    def simulation_step(self):
        """Advance the simulator by one step."""
        self.nr3.solve()

    def simulation_command(self, command):
        """Run an custom command on simulator."""
        self.nr3 = NR3(command.split()[1].strip('"'))
        self.all_bus_name = self.nr3.all_bus_names
        self.offsets = {}
        for k, v in enumerate(self.nr3.all_node_names):
            self.offsets[v] = k
        self.loads = {}
        self.load_to_bus = {}
        for load in self.get_node_ids():
            dss.Loads.Name(load)
            bus_phase = dss.CktElement.BusNames()[0].split('.')
            if len(bus_phase) == 1:
                bus_phase.extend(['1','2','3'])
            self.loads[load] = [['.'.join([bus_phase[0], i]) for i in bus_phase[1:] if i != '0'], dss.CktElement.NumPhases()]
            self.load_to_bus[load] = bus_phase[0]

    def set_solution_mode(self, value):
        """Set solution mode on simulator."""
        pass

    def set_solution_number(self, value):
        """Set solution number on simulator."""
        pass

    def set_solution_step_size(self, value):
        """Set solution stepsize on simulator."""
        pass

    def set_solution_control_mode(self, value):
        """Set solution control mode on simulator."""
        pass

    def set_solution_max_control_iterations(self, value):
        """Set solution max control iterations on simulator."""
        pass

    def set_solution_max_iterations(self, value):
        """Set solution max iternations on simulator."""
        pass

    def check_simulation_converged(self):
        """Check if the solver has converged."""
        return self.nr3.converged

    def get_node_ids(self):
        """Get list of node ids."""
        nodes = self.nr3.all_load_names
        return nodes

    def update_all_bus_voltages(self):
        self.puvoltage = self.nr3.get_all_bus_voltages() #work around this

    def get_node_voltage(self, node_id):
        puvoltage = 0 # get rid of this
        for phase in range(self.loads[node_id][1]):
            puvoltage += self.puvoltage[self.offsets[self.loads[node_id][0][phase]]]
        puvoltage /= self.loads[node_id][1]
        return puvoltage

    def get_total_power(self):
        return 0

    def get_losses(self):
        return 0

    def set_node_kw(self, node_id, value):
        """Set node kW."""
        self.nr3.set_load_kw(node_id, value)

    def set_node_kvar(self, node_id, value):
        """Set node kVar."""
        self.nr3.set_load_kvar(node_id, value)

    def set_slack_bus_voltage(self, value):
        """Set slack bus voltage."""
        pass

    # ######################## REGULATOR ############################
    def get_all_regulator_names(self):
        pass



    def set_regulator_property(self, reg_id, prop):
        pass

    def get_regulator_tap(self, reg_id):
        return 0

    def get_regulator_forwardband(self, reg_id):
        return 0

    def get_regulator_forwardvreg(self, reg_id):
        return 0

    def get_substation_top_voltage(self):
        return 0

    def get_substation_bottom_voltage(self):
        return 0

    def get_worst_u_node(self):
        u_all = []
        v_all = {}
        u_all_real = {}
        u_worst = 0
        for bus in self.all_bus_name:
            try:
                va = self.puvoltage[self.offsets['{}.{}'.format(bus, 1)]]
            except:
                va = 0
            try:
                vb = self.puvoltage[self.offsets['{}.{}'.format(bus, 2)]]
            except:
                vb = 0
            try:    
                vc = self.puvoltage[self.offsets['{}.{}'.format(bus, 3)]]
            except:
                vc = 0

            v_all[bus] = [va, vb, vc]
            mean = (va + vb + vc) / 3
            max_diff = max(abs(va - mean), abs(vb - mean), abs(vc - mean))
            u = max_diff / mean
            if u > u_worst:
                u_worst = u
                v_worst = [va, vb, vc]
            u_all.append(u)
            u_all_real[bus] = u

        return u_worst, v_worst, np.mean(u_all), np.std(u_all), v_all, u_all_real, self.load_to_bus
