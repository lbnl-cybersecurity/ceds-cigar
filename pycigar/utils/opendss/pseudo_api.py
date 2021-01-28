"""Contains the Power/opendss API manager."""
import opendssdirect as dss
import numpy as np
import warnings
from pycigar.utils.logging import logger
import time

SBASE = 1000000.0

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
        #start_time = time.time()
        """Advance the simulator by one step."""
        dss.Solution.Solve()
        #print(time.time() - start_time)

    def simulation_command(self, command):
        """Run an custom command on simulator."""
        dss.run_command(command)
        self.all_bus_name = dss.Circuit.AllBusNames()

        self.offsets = {}
        for k, v in enumerate(dss.Circuit.AllNodeNames()):
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

        self.current_offsets = {}
        pos = 0
        for pd in dss.PDElements.AllNames():
            dss.Circuit.SetActiveElement(pd)
            pd_type, pd_name = pd.split('.')
            if pd_type == 'Line':
                self.current_offsets[pd_name] = [pos, dss.CktElement.NumPhases()]
            pos += len(dss.CktElement.CurrentsMagAng())

        self.ibase = {}
        for line in self.current_offsets.keys():
            dss.Lines.Name(line)
            bus = dss.Lines.Bus1()
            dss.Circuit.SetActiveBus(bus)
            IBase = SBASE/(dss.Bus.kVBase()*1000)
            self.ibase[line] = IBase

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
        output = dss.Solution.Converged
        if output is False:
            warnings.warn('OpenDSS does not converge.')
        return output

    def get_node_ids(self):
        """Get list of node ids."""
        nodes = dss.Loads.AllNames()
        return nodes

    def update_all_bus_voltages(self):
        self.puvoltage = dss.Circuit.AllBusMagPu()

    def get_all_currents(self):
        self.currents = dss.PDElements.AllCurrentsMagAng()
        result = {}
        for line in self.current_offsets:
            #result[line] = np.array(self.currents[self.current_offsets[line][0]:self.current_offsets[line][0] + self.current_offsets[line][1]*2:2])/self.ibase[line]
            result[line] = self.currents[self.current_offsets[line][0]:self.current_offsets[line][0] + self.current_offsets[line][1]*2:2]

        return result

    def get_node_voltage(self, node_id):
        puvoltage = 0 # get rid of this
        for phase in range(self.loads[node_id][1]):
            puvoltage += self.puvoltage[self.offsets[self.loads[node_id][0][phase]]]
        puvoltage /= self.loads[node_id][1]
        return puvoltage

    def get_total_power(self):
        return np.array(dss.Circuit.TotalPower())

    def get_losses(self):
        return np.array(dss.Circuit.Losses())

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
        return dss.RegControls.AllNames()


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
        #voltage_a = self.get_node_voltage('s701a')
        #voltage_b = self.get_node_voltage('s701b')
        #voltage_c = self.get_node_voltage('s701c')
        #voltage = (voltage_a + voltage_b + voltage_c) / 3
        #return voltage
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

            u = 0
            if va != 0 and vb != 0 and vc != 0:
                mean = (va + vb + vc) / 3
                max_diff = max(abs(va - mean), abs(vb - mean), abs(vc - mean))
                u = max_diff / mean
            if u > u_worst:
                u_worst = u
                v_worst = [va, vb, vc]
            u_all.append(u)
            u_all_real[bus] = u

        return u_worst, v_worst, np.mean(u_all), np.std(u_all), v_all, u_all_real, self.load_to_bus
