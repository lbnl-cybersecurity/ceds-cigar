"""Contains the Power/opendss API manager."""
import numpy as np
import opendssdirect as dss


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
        self.dss = dss

    def simulation_step(self):
        """Advance the simulator by one step."""
        self.dss.Solution.Solve()

    def simulation_command(self, command):
        """Run an custom command on simulator."""
        self.dss.run_command(command)

    def set_solution_mode(self, value):
        """Set solution mode on simulator."""
        self.dss.Solution.Mode(value)

    def set_solution_number(self, value):
        """Set solution number on simulator."""
        self.dss.Solution.Number(value)

    def set_solution_step_size(self, value):
        """Set solution stepsize on simulator."""
        self.dss.Solution.StepSize(value)

    def set_solution_control_mode(self, value):
        """Set solution control mode on simulator."""
        self.dss.Solution.ControlMode(value)

    def set_solution_max_control_iterations(self, value):
        """Set solution max control iterations on simulator."""
        self.dss.Solution.MaxControlIterations(value)

    def set_solution_max_iterations(self, value):
        """Set solution max iternations on simulator."""
        self.dss.Solution.MaxIterations(value)

    def check_simulation_converged(self):
        """Check if the solver has converged."""
        output = self.dss.Solution.Converged
        return output

    def get_node_ids(self):
        """Get list of node ids."""
        nodes = self.dss.Loads.AllNames()
        return nodes

    def get_regulator_ids(self):
        regs = self.dss.RegControls.AllNames()
        return regs

    def get_node_voltage(self, node_id):
        """Get node voltage given node id."""
        self.dss.Loads.Name(node_id)
        voltage = self.dss.CktElement.VoltagesMagAng()
        voltage = (voltage[0] + voltage[2] + voltage[4]) / (
                    self.dss.CktElement.NumPhases() * (self.dss.Loads.kV() * 1000 / (3 ** 0.5)))
        # get the pu information directly
        if np.isnan(voltage) or np.isinf(voltage):
            raise ValueError('Voltage Output {} from OpenDSS for Load {} at Bus {} is not appropriate.'.
                             format(np.mean(voltage), node_id, self.dss.CktElement.BusNames()[0]))
        else:
            output = np.mean(voltage)
        return output

    def set_node_kw(self, node_id, value):
        """Set node kW."""
        self.dss.Loads.Name(node_id)
        self.dss.Loads.kW(value)

    def set_node_kvar(self, node_id, value):
        """Set node kVar."""
        self.dss.Loads.Name(node_id)
        self.dss.Loads.kvar(value)

    def set_slack_bus_voltage(self, value):
        """Set slack bus voltage."""
        self.dss.Vsources.PU(value)

    def set_regulator_property(self, reg_name, prop: dict):
        self.dss.RegControls.Name(reg_name)
        for k, v in prop.items():
            if k == 'max_tap_change':
                self.dss.RegControls.MaxTapChange(v)
            elif k == "forward_band":
                self.dss.RegControls.ForwardBand(v)
            elif k == 'tap_number':
                self.dss.RegControls.TapNumber(v)
            elif k == 'tap_delay':
                self.dss.RegControls.TapDelay(v)
            else:
                print('Regulator Update Parameters not understood (pseudo_api.py)')

    def set_line_property(self, line_name, prop: dict):
        if 'status' in prop.keys():
            if prop['status'] == 1:
                self.dss.Text.Command('close line.{} term=1'.format(line_name))
            elif prop['status'] == 0:
                self.dss.Text.Command('open line.{} term=1'.format(line_name))
            else:
                raise SystemError('Line Status for line {} not understood, status should be 1/0'.format(line_name))
        else:
            raise SystemError('No other key except "status" is accepted for line {}'.format(line_name))
