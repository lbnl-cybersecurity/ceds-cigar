# A sample code to test the python OpenDSS interface
from DSSStartup import DSSStartup
from setInfo import *
from getInfo import *
import numpy as np


# DSSStartup returns a dictionary
result= DSSStartup()
dssText=result['dsstext']
dssSolution=result['dsssolution']
dssCircuit=result['dsscircuit']
DSSObj=result['dssobj']
# Compile the circuit
dssText.Command="Compile C:/feeders/feeder13_B_R/feeder13BR.dss "
# Run a Power Flow Solution
dssSolution.Solve()


# Get Values for two specific loads, the getloadinfo returns a list of dictionary objects
LoadInfo=getLoadInfo(DSSObj,['load_645','load_611'])
#  Getting the real power of the loads
LoadRealPower=[d['kw'] for d in LoadInfo]
# List does not support arithmatic, so you can convert it to an array
LoadRealPowerArray=np.asarray(LoadRealPower)


#  Get Values for all loads
value=getLoadInfo(DSSObj,[])
LoadRealPower=[d['kw'] for d in value]
# List does not support arithmatic, so you can convert it to an array
LoadRealPowerArray=np.asarray(LoadRealPower)
LoadNames=[d['name'] for d in value]

# Change the load power by a factor of 2 and then rerun a simulation
LoadRealPowerArray= 2 * LoadRealPowerArray
setLoadInfo(DSSObj,LoadNames,'kw',LoadRealPowerArray.tolist(),0)
dssSolution.Solve()
# print(value)

# Get Incoming Flow from a line
LineInfo=getLineInfo(DSSObj,['L_U_650'])
bus1power=[d['bus1powerreal'] for d in LineInfo]
print(bus1power)

# Get some information about the buses 
BusInfo=getBusInfo(DSSObj,['bus_634','bus_645'])
print(BusInfo)
