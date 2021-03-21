from pycigar.core.kernel.simulation import KernelSimulation
import os

# import time
# import logging
# import subprocess
import signal
from pycigar.utils.opendss.pseudo_api import PyCIGAROpenDSSAPI
import socket
from contextlib import closing

RETRIES_ON_ERROR = 10


class OpenDSSSimulation(KernelSimulation):
    def __init__(self, master_kernel):
        """See parent class."""
        KernelSimulation.__init__(self, master_kernel)

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def start_simulation(self):
        """See parent class."""
        # This is an example to open a new simulator process.
        # set port to random available port
        # port = self.find_free_port()
        # opendss_call = "/home/toanngo/Documents/GitHub/power/power
        # /utils/opendss/opendss_server.py"
        # self.opendss_proc = subprocess.Popen(["python", opendss_call,
        # "--port={}".format(port)], preexec_fn=os.setsid)
        return PyCIGAROpenDSSAPI()

    def update(self, reset):
        """See parent class."""
        self.kernel_api.simulation_step()

    def close(self):
        """See parent class."""
        # self.teardown_opendss()
        pass

    def check_converged(self):
        """See parent class."""
        return self.kernel_api.check_simulation_converged()
