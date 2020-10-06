"""Contains the Power/opendss API manager."""
import opendssdirect as dss
import numpy as np
import warnings
from pycigar.utils.logging import logger
import time

class PyCIGAROpenDSSAPI(object):
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
        dss.Solution.Solve()

    def simulation_command(self, command):
        """Run an custom command on simulator."""
        dss.run_command(command)
        self.all_bus_name = dss.Circuit.AllBusNames()

        self.offsets = {}
        for k, v in enumerate(dss.Circuit.AllNodeNames()):
            self.offsets[v] = k
        self.loads = {}
        for load in self.get_node_ids():
            dss.Loads.Name(load)
            bus_phase = dss.CktElement.BusNames()[0].split('.')
            if len(bus_phase) == 1:
                bus_phase.extend(['1','2','3'])
            self.loads[load] = [['.'.join([bus_phase[0], i]) for i in bus_phase[1:] if i != '0'], dss.CktElement.NumPhases()]

    def set_solution_mode(self, value):
        """Set solution mode on simulator."""
        dss.Solution.Mode(value)

    def set_solution_number(self, value):
        """Set solution number on simulator."""
        dss.Solution.Number(value)

    def set_solution_step_size(self, value):
        """Set solution stepsize on simulator."""
        dss.Solution.StepSize(value)

    def set_solution_control_mode(self, value):
        """Set solution control mode on simulator."""
        dss.Solution.ControlMode(value)

    def set_solution_max_control_iterations(self, value):
        """Set solution max control iterations on simulator."""
        dss.Solution.MaxControlIterations(value)

    def set_solution_max_iterations(self, value):
        """Set solution max iternations on simulator."""
        dss.Solution.MaxIterations(value)

    def check_simulation_converged(self):
        """Check if the solver has converged."""
        output = dss.Solution.Converged()
        if output is False:
            warnings.warn('OpenDSS does not converge.')
        return output

    def get_node_ids(self):
        """Get list of node ids."""
        nodes = dss.Loads.AllNames()
        return nodes

    def update_all_bus_voltages(self):
        self.puvoltage = dss.Circuit.AllBusMagPu() #work around this

    def get_node_voltage(self, node_id):
        puvoltage = 0 # get rid of this
        for phase in range(self.loads[node_id][1]):
            puvoltage += self.puvoltage[self.offsets[self.loads[node_id][0][phase]]]
        puvoltage /= self.loads[node_id][1]
        return puvoltage

    def get_total_power(self):
        start_time = time.time()
        result = np.array(dss.Circuit.TotalPower())
        if 'opendss_time' not in logger().custom_metrics:
            logger().custom_metrics['opendss_time'] = 0
        logger().custom_metrics['opendss_time'] += time.time() - start_time

    def get_losses(self):
        result = np.array(dss.Circuit.Losses())

    def set_node_kw(self, node_id, value):
        """Set node kW."""
        dss.Loads.Name(node_id)
        dss.Loads.kW(value)

    def set_node_kvar(self, node_id, value):
        """Set node kVar."""
        dss.Loads.Name(node_id)
        dss.Loads.kvar(value)

    def set_slack_bus_voltage(self, value):
        """Set slack bus voltage."""
        dss.Vsources.PU(value)

    # ######################## REGULATOR ############################
    def get_all_regulator_names(self):
        result = dss.RegControls.AllNames()


    def set_regulator_property(self, reg_id, prop):
        dss.RegControls.Name(reg_id)
        for k, v in prop.items():
            if v is not None:
                v = int(v)
                if k == 'max_tap_change':
                    dss.RegControls.MaxTapChange(v)
                elif k == "forward_band":
                    dss.RegControls.ForwardBand(v)
                elif k == 'tap_number':
                    dss.RegControls.TapNumber(v)
                elif k == 'tap_delay':
                    dss.RegControls.TapDelay(v)
                elif k =='delay':
                    dss.RegControls.Delay(v)
                else:
                    print('Regulator Parameters unknown by PyCIGAR. Checkout pycigar/utils/opendss/pseudo_api.py')

    def get_regulator_tap(self, reg_id):
        dss.RegControls.Name(reg_id)
        return dss.RegControls.TapNumber()

    def get_regulator_forwardband(self, reg_id):
        dss.RegControls.Name(reg_id)
        return dss.RegControls.ForwardBand()

    def get_regulator_forwardvreg(self, reg_id):
        dss.RegControls.Name(reg_id)
        return dss.RegControls.ForwardVreg()

    def get_substation_top_voltage(self):
        sourcebus = self.all_bus_name[0]
        num_phases = 0
        voltage = 0
        for i in range(3):
            num_phases += 1
            voltage += self.puvoltage[self.offsets['{}.{}'.format(sourcebus, i+1)]]
        voltage /= num_phases
        return voltage

    def get_substation_bottom_voltage(self):
        voltage_a = self.get_node_voltage('s701a')
        voltage_b = self.get_node_voltage('s701b')
        voltage_c = self.get_node_voltage('s701c')
        voltage = (voltage_a + voltage_b + voltage_c) / 3
        return voltage


    def get_worst_u_node(self):
        buses = dss.Circuit.AllBusNames()
        u_all = []
        v_all = {}
        u_all_real = {}
        u_worst = 0
        for bus in self.all_bus_name:
            va = self.puvoltage[self.offsets['{}.{}'.format(bus, 1)]]
            vb = self.puvoltage[self.offsets['{}.{}'.format(bus, 2)]]
            vc = self.puvoltage[self.offsets['{}.{}'.format(bus, 3)]]

            v_all[bus] = [va, vb, vc]
            mean = (va + vb + vc) / 3
            max_diff = max(abs(va - mean), abs(vb - mean), abs(vc - mean))
            u = max_diff / mean
            if u > u_worst:
                u_worst = u
                v_worst = [va, vb, vc]
            u_all.append(u)
            u_all_real[bus] = u

        return u_worst, v_worst, np.mean(u_all), np.std(u_all), v_all, u_all_real