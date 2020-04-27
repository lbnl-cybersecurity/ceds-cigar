import numpy as np
import socket
from _thread import start_new_thread
import struct
import pycigar.utils.opendss.constants as commands

import opendssdirect as dss
import argparse

parser = argparse.ArgumentParser(description='Create OpenDSS Server.')
parser.add_argument('--port', action='store', default=9999, type=int,
                    help='bind server to port')

PORT = parser.parse_args().port


def send_message(conn, in_format, values):
    """Send a message to the client.

    If the message is a string, it is sent in segments of length 256 (if the
    string is longer than such) and concatenated on the client end.

    Parameters
    ----------
    conn : socket.socket
        socket for server connection
    in_format : str
        format of the input structure
    values : tuple of Any
        commands to be encoded and issued to the client
    """
    if in_format == 'str':
        packer = struct.Struct(format='i')
        values = values[0]

        # when the message is too large, send value in segments and inform the
        # client that additional information will be sent. The value will be
        # concatenated on the other end
        while len(values) > 256:
            # send the next set of data
            conn.send(values[:256].encode())
            values = values[256:]

            # wait for a reply
            data = None
            while data is None:
                data = conn.recv(2048)

            # send a not-done signal
            packed_data = packer.pack(*(1,))
            conn.send(packed_data)

        # send the remaining components of the message (which is of length less
        # than or equal to 256)
        conn.send(values.encode())

        # wait for a reply
        data = None
        while data is None:
            data = conn.recv(2048)

        # send a done signal
        packed_data = packer.pack(*(0,))
        conn.send(packed_data)
    else:
        packer = struct.Struct(format=in_format)
        packed_data = packer.pack(*values)
        conn.send(packed_data)


def retrieve_message(conn, out_format):
    """Retrieve a message from the client.

    Parameters
    ----------
    conn : socket.socket
        socket for server connection
    out_format : str or None
        format of the output structure

    Returns
    -------
    Any
        received message
    """
    unpacker = struct.Struct(format=out_format)
    try:
        data = conn.recv(unpacker.size)
        unpacked_data = unpacker.unpack(data)
    finally:
        pass
    return unpacked_data


def threaded_client(conn):
    # send feedback that the connection is active
    conn.send('Ready.'.encode())

    done = False
    while not done:
        data = conn.recv(256).decode()

        if data is not None:
            if data == '':
                continue

        # convert to integer, this is the command
        data = int(data)

        if data == commands.SIMULATION_STEP:
            send_message(conn, in_format='i', values=(0,))
            dss.Solution.Solve()

        elif data == commands.SIMULATION_COMMAND:
            send_message(conn, in_format='i', values=(0,))
            data = None
            while data is None:
                data = conn.recv(2048)

            command = data.decode("utf-8")
            dss.run_command(command)

        elif data == commands.SET_SOLUTION_MODE:
            send_message(conn, in_format='i', values=(0,))
            data = retrieve_message(conn, out_format='i')
            dss.Solution.Mode(data)

        elif data == commands.SET_SOLUTION_NUMBER:
            send_message(conn, in_format='i', values=(0,))
            data = retrieve_message(conn, out_format='i')
            dss.Solution.Number(data)

        elif data == commands.SET_SOLUTION_STEP_SIZE:
            send_message(conn, in_format='i', values=(0,))
            data = retrieve_message(conn, out_format='i')
            dss.Solution.StepSize(data)

        elif data == commands.SET_SOLUTION_CONTROL_MODE:
            send_message(conn, in_format='i', values=(0,))
            data = retrieve_message(conn, out_format='i')
            dss.Solution.ControlMode(data)

        elif data == commands.SET_SOLUTION_MAX_CONTROL_ITERATIONS:
            send_message(conn, in_format='i', values=(0,))
            data = retrieve_message(conn, out_format='i')
            dss.Solution.MaxControlIterations(data)

        elif data == commands.SET_SOLUTION_MAX_ITERATIONS:
            send_message(conn, in_format='i', values=(0,))
            data = retrieve_message(conn, out_format='i')
            dss.Solution.MaxIterations(data)

        elif data == commands.CHECK_SIMULATION_CONVERGED:
            send_message(conn, in_format='i', values=(0,))
            output = dss.Solution.Converged
            send_message(conn, in_format='?', values=(output,))

        elif data == commands.GET_NODE_IDS:
            send_message(conn, in_format='i', values=(0,))
            nodes = dss.Loads.AllNames()
            if len(nodes) == 0:
                output = '-1'
            else:
                output = ':'.join([str(n) for n in nodes])
            send_message(conn, in_format='str', values=(output,))

        elif data == commands.SET_NODE_KW:
            send_message(conn, in_format='i', values=(0,))
            data = None
            while data is None:
                data = conn.recv(1024)
            command = data.decode("utf-8")
            node_id, value = command.split()
            value = float(value)
            dss.Loads.Name(node_id)
            dss.Loads.kW(value)

        elif data == commands.SET_NODE_KVAR:
            send_message(conn, in_format='i', values=(0,))
            data = None
            while data is None:
                data = conn.recv(2048)
            command = data.decode("utf-8")
            node_id, value = command.split()
            value = float(value)
            dss.Loads.Name(node_id)
            dss.Loads.kvar(value)

        elif data == commands.GET_NODE_VOLTAGE:
            send_message(conn, in_format='i', values=(0,))
            data = None
            while data is None:
                data = conn.recv(2048)
            node_id = data.decode("utf-8")
            dss.Circuit.SetActiveElement('load.' + node_id)  # set active element
            dss.Circuit.SetActiveBus(dss.CktElement.BusNames()[0])  # grab the bus for the active element
            voltage = dss.Bus.puVmagAngle()[::2]  # get the pu information directly
            if (np.isnan(np.mean(voltage)) or np.isinf(np.mean(voltage))):
                raise ValueError('Voltage Output {} from OpenDSS for Load {} at Bus {} is not appropriate.'.
                                 format(np.mean(voltage), node_id, dss.CktElement.BusNames()[0]))
            else:
                output = np.mean(voltage)
                send_message(conn, in_format='f', values=(output,))


while True:
    # tcp/ip connection from the opendss process
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', PORT))

    # connect to the Power instance
    server_socket.listen(10)
    c, address = server_socket.accept()

    # start the threaded process
    start_new_thread(threaded_client, (c,))
