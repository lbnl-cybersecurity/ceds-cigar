"""Constants used by the OpenDSS API for sending/receiving TCP messages."""
"""opendssdirect documentation: http://dss-extensions.org/OpenDSSDirect.py/opendssdirect.html"""

###############################################################################
#                             Simulation Commands                             #
###############################################################################

#: simulation step
SIMULATION_STEP = 0x00

#: reset the simulation
SIMULATION_RESET = 0x01

#: reset the simulation
CHECK_SIMULATION_CONVERGED = 0x02

#: set present solution mode
SET_SOLUTION_MODE = 0x03

#: set present solution number
SET_SOLUTION_NUMBER = 0x04

#: set present solution step size
SET_SOLUTION_STEP_SIZE = 0x05

#: set present solution control mode
SET_SOLUTION_CONTROL_MODE = 0x06

#: set present solution max control iterations
SET_SOLUTION_MAX_CONTROL_ITERATIONS = 0x07

#: set present solution max iterations
SET_SOLUTION_MAX_ITERATIONS = 0x08

#: run arbitrary command
SIMULATION_COMMAND = 0x09

###############################################################################
#                             Scenarios Commands                              #
###############################################################################
GET_NODE_IDS = 0x0A

###############################################################################
#                             Node Commands                                   #
###############################################################################
SET_NODE_KW = 0x0B

SET_NODE_KVAR = 0x0C

GET_NODE_VOLTAGE = 0x0D
