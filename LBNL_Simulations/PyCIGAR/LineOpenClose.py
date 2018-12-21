'''
This code is an example of how to open and close lines in OpenDSS using the DSSText Command.
It is better not to go through the setlineinfo functions, rather doing this using the DSSText Command 
'''

# Importing the libraries
from utils.openDSS.DSSStartup import DSSStartup
from utils.openDSS.setInfo import *
from utils.openDSS.getInfo import *
import numpy as np
import matplotlib.pyplot as plt


# DSSStartup returns a dictionary
result= DSSStartup()
DSSText=result['dsstext']
DSSSolution=result['dsssolution']
DSSCircuit=result['dsscircuit']
DSSObj=result['dssobj']
# Compile the circuit
DSSText.Command="Compile C:/feeders/feeder33_Meshed/33BusRadial.dss"

DSSSolution.Solve()

DSSCircuit.SetActiveElement('line_30')
print('Flow Before Opening  line 29 and 5: '+ str(DSSCircuit.ActiveElement.Powers[:2]))

# Opening two lines; this is the easiest way of doing this without going through the setting function
# remember the enabled feature does not work as it should be 
# Follow this discussion form if required : https://sourceforge.net/p/electricdss/discussion/861976/thread/7c3a789530/ 
DSSText.command = 'open line.line_29 term=1'
DSSText.command = 'open line.line_5 term=1'

DSSSolution.Solve()
DSSCircuit.SetActiveElement('line_30')
print('Flow After Opening  line 29 and 5: '+ str(DSSCircuit.ActiveElement.Powers[:2]))
# The values found here are close because the network is weakly mshed; in case of a radial network the branches underlying will have zero current

#  Close the two lines 
DSSText.command = 'close line.line_29 term=1'
DSSText.command = 'close line.line_5 term=1'

# Check whether the flows restored
DSSSolution.Solve()
DSSCircuit.SetActiveElement('line_30')
print('Flow After Opening  line 29 and 5: '+ str(DSSCircuit.ActiveElement.Powers[:2]))