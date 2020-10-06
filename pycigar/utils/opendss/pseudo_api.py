"""Contains the Power/opendss API manager."""
import opendssdirect as dss
import numpy as np
import warnings

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

    def get_node_voltage(self, node_id):
        """Get node voltage given node id."""
        dss.Loads.Name(node_id)
        _voltage = dss.CktElement.VoltagesMagAng()
        # print(voltage, node_id, dss.CktElement.NumPhases())
        if len(_voltage) == 6 or len(_voltage) == 8:
            voltage = (_voltage[0] + _voltage[2] + _voltage[4]) / (dss.CktElement.NumPhases() * (dss.Loads.kV() * 1000 / (3**0.5)))
        else:
            if node_id[-1] == 'a':
                voltage = (_voltage[0]) / (dss.CktElement.NumPhases() * (dss.Loads.kV() * 1000 / (3 ** 0.5)))
            elif node_id[-1] == 'b':
                voltage = (_voltage[0]) / (dss.CktElement.NumPhases() * (dss.Loads.kV() * 1000 / (3 ** 0.5)))
            else:
                voltage = (_voltage[0]) / (dss.CktElement.NumPhases() * (dss.Loads.kV() * 1000 / (3 ** 0.5)))

        # get the pu information directly
        if np.isnan(voltage) or np.isinf(voltage):
            print('NumPhases', dss.CktElement.NumPhases())
            print('kV', dss.Loads.kV())
            print('voltage', _voltage)
            raise ValueError(
                'Voltage Output {} from OpenDSS for Load {} at Bus {} is not appropriate.'.format(
                    np.mean(voltage), node_id, dss.CktElement.BusNames()[0]
                )
            )
        else:
            output = np.mean(voltage)
        return output

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
        pass

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
        dss.Bus.Name()
        voltage = dss.Bus.VMagAngle()
        voltage = (voltage[0] + voltage[2] + voltage[4]) / (
            dss.CktElement.NumPhases() * (dss.Loads.kV() * 1000 / (3 ** 0.5))
        )
        return voltage

    def get_substation_bottom_voltage(self):
        dss.Loads.Name('S701a')
        voltage_a = dss.CktElement.VoltagesMagAng()
        dss.Loads.Name('S701b')
        voltage_b = dss.CktElement.VoltagesMagAng()
        dss.Loads.Name('S701c')
        voltage_c = dss.CktElement.VoltagesMagAng()
        voltage = (voltage_a[0] + voltage_b[0] + voltage_c[0]) / (3 * (dss.Loads.kV() * 1000 / (3 ** 0.5)))
        return voltage

    def get_worst_u_node(self):
        buses = dss.Circuit.AllBusNames()
        u_all = []
        v_all = {}
        u_all_bus = {}
        u_worst = 0
        for bus in buses:
            dss.Circuit.SetActiveBus(bus)
            vmagang = dss.Bus.puVmagAngle()
            va = vmagang[0]
            vb = vmagang[2]
            vc = vmagang[4]
            v_all[bus] = [va, vb, vc]
            mean = (va + vb + vc) / 3
            max_diff = max(abs(va - mean), abs(vb - mean), abs(vc - mean))
            u = max_diff / mean
            if u > u_worst:
                u_worst = u
                v_worst = [va, vb, vc]
            u_all.append(u)
            u_all_bus[bus] = u

        return u_worst, v_worst, np.mean(u_all), np.std(u_all), v_all, u_all_bus

    def get_worst_u_node_real(self):
        buses = dss.Circuit.AllBusNames()
        u_all = []
        v_all = {}
        u_all_real = {}
        u_worst = 0
        for bus in buses:
            dss.Circuit.SetActiveBus(bus)
            vmagang = dss.Bus.puVmagAngle()
            va = vmagang[0]
            vb = vmagang[2]
            vc = vmagang[4]
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

