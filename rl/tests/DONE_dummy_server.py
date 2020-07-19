import socket
from _thread import start_new_thread
import struct
import power.utils.opendss.constants as commands

import opendssdirect as dss
import argparse

parser = argparse.ArgumentParser(description='Create OpenDSS Server.')
parser.add_argument('--port', action='store', default=9999, type=int,
                   help='bind server to port')

PORT = parser.parse_args().port

#import opendssdirect as dss
#OpenDSSDirectory = '/home/toanngo/Documents/GitHub/power/tests/feeder34_test.dss'
#dss.run_command('Redirect ' + OpenDSSDirectory)
#dss.Solution.MaxControlIterations(1000000)
#print(dss.Loads.Count())

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
    #send feedback that the connection is active
    conn.send('Ready.'.encode())

    done = False
    while not done:
        data = conn.recv(256).decode()

        if data is not None:
            if data == '':
                continue

        #convert to integer, this is the command
        data = int(data)

        if data == commands.GET_NODES_NAME:
            send_message(conn, in_format='i', values=(0,))

            #run opendss command here and receive the result is data
            data = "result data to be sent to client"
            print(data, file=sys.stderr)

            send_message(conn, in_format='str', values=(data,))


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
