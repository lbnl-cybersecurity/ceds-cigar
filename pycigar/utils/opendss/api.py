"""Contains the Power/opendss API manager."""

import socket
import struct
import pycigar.utils.opendss.constants as commands


def create_client(port, print_status=False):
    """Create a socket connection with the server.

    Parameters
    ----------
    port : int
        the port number of the socket connection
    print_status : bool, optional
        specifies whether to print a status check while waiting for connection
        between the server and client
    Returns
    -------
    socket.socket
        socket for client connection
    """
    # create a socket connection
    if print_status:
        print('Listening for connection...', end=' ')

    stop = False
    while not stop:
        # try to connect
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('localhost', port))

            # check the connection
            data = None
            while data is None:
                data = s.recv(2048)
            stop = True

        # except Exception as e:
        #   logging.debug('Cannot connect to the server: {}'.format(e))

        except socket.error:
            stop = False

    # print the return statement
    if print_status:
        print(data.decode('utf-8'))

    return s


class PowerOpenDSSAPI(object):
    """An API used to interact with OpenDSS via a TCP connection."""

    def __init__(self, port):
        """Instantiate the API.

        Parameters
        ----------
        port : int
            the port number of the socket connection
        """
        self.port = port
        self.s = create_client(port, print_status=True)

    def _send_command(self, command_type, in_format, values, out_format):
        """Send an arbitrary command via the connection.

        Commands are sent in two stages. First, the client sends the command
        type (e.g. ac.REMOVE_VEHICLE) and waits for a conformation message from
        the server. Once the confirmation is received, the client send a
        encoded binary packet that the server will be prepared to decode, and
        will then receive some return value (either the value the client was
        requesting or a 0 signifying that the command has been executed. This
        value is then returned by this method.

        Parameters
        ----------
        command_type : flow.utils.opendss.constants.*
            the command the client would like OpenDSS to execute
        in_format : str or None
            format of the input structure
        values : tuple of Any or None
            commands to be encoded and issued to the server
        out_format : str or None
            format of the output structure

        Returns
        -------
        Any
            the final message received from the OpenDSS server
        """
        # send the command type to the server
        self.s.send(str(command_type).encode())

        # wait for a response
        unpacker = struct.Struct(format='i')
        data = None
        while data is None:
            data = self.s.recv(unpacker.size)

        # send the command values
        if in_format is not None:
            if in_format == 'str':
                self.s.send(str.encode(values))
            else:
                packer = struct.Struct(format=in_format)
                packed_data = packer.pack(*values)
                self.s.send(packed_data)
        else:
            # if no command is needed, just send a status response
            self.s.send(str.encode('1'))

        # collect the return values
        if out_format is not None:
            if out_format == 'str':
                done = False
                unpacked_data = ''
                while not done:
                    # get the next bit of data
                    data = None
                    while data is None or data == b'':
                        data = self.s.recv(256)

                    # concatenate the results
                    unpacked_data += data.decode('utf-8')

                    # ask for a status check (just by sending any command)
                    self.s.send(str.encode('1'))

                    # check if done
                    unpacker = struct.Struct(format='i')
                    data = None
                    while data is None:
                        data = self.s.recv(unpacker.size)
                    done = unpacker.unpack(data)[0] == 0
            else:
                unpacker = struct.Struct(format=out_format)
                data = None
                while data is None:
                    data = self.s.recv(unpacker.size)
                unpacked_data = unpacker.unpack(data)

            return unpacked_data

    def simulation_step(self):
        """Advance the simulator by one step."""
        self._send_command(commands.SIMULATION_STEP, in_format=None, values=None, out_format=None)

    def simulation_command(self, command):
        """Run an custom command on simulator."""
        self._send_command(commands.SIMULATION_COMMAND, in_format='str', values=command, out_format=None)

    def set_solution_mode(self, value):
        """Set solution mode on simulator."""
        self._send_command(commands.SET_SOLUTION_MODE, in_format='i', values=value, out_format=None)

    def set_solution_number(self, value):
        """Set solution number on simulator."""
        self._send_command(commands.SET_SOLUTION_NUMBER, in_format='i', values=value, out_format=None)

    def set_solution_step_size(self, value):
        """Set solution stepsize on simulator."""
        self._send_command(commands.SET_SOLUTION_STEP_SIZE, in_format='i', values=value, out_format=None)

    def set_solution_control_mode(self, value):
        """Set solution control mode on simulator."""
        self._send_command(commands.SET_SOLUTION_CONTROL_MODE, in_format='i', values=value, out_format=None)

    def set_solution_max_control_iterations(self, value):
        """Set solution max control iterations on simulator."""
        self._send_command(commands.SET_SOLUTION_MAX_CONTROL_ITERATIONS, in_format='i', values=value, out_format=None)

    def set_solution_max_iterations(self, value):
        """Set solution max iternations on simulator."""
        self._send_command(commands.SET_SOLUTION_MAX_ITERATIONS, in_format='i', values=value, out_format=None)

    def check_simulation_converged(self):
        """Check if the solver has converged."""
        (output,) = self._send_command(commands.CHECK_SIMULATION_CONVERGED, in_format=None, values=None, out_format='?')
        return output

    def get_node_ids(self):
        """Get list of node ids."""
        response = self._send_command(commands.GET_NODE_IDS, in_format=None, values=None, out_format='str')
        return response.split(':')

    def get_node_voltage(self, node_id):
        """Get node voltage given a node id."""
        (response,) = self._send_command(commands.GET_NODE_VOLTAGE, in_format='str', values=node_id, out_format='f')
        return response

    def set_node_kw(self, node_id, value):
        """Set node kW."""
        command = node_id + " " + str(value)
        self._send_command(commands.SET_NODE_KW, in_format='str', values=command, out_format=None)

    def set_node_kvar(self, node_id, value):
        """Set node kVAR."""
        command = node_id + " " + str(value)
        self._send_command(commands.SET_NODE_KVAR, in_format='str', values=command, out_format=None)
