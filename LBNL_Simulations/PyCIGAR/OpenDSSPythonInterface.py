# A sample code to test the python OpenDSS interface
from utils.openDSS.DSSStartup import DSSStartup
from utils.openDSS.setInfo_linux import *
from utils.openDSS.getInfo_linux import *
import opendssdirect as DSS
#import numpy as np


# DSSStartup returns a dictionary
DSS.run_command('Redirect C:\\feeders\\feeder33_Meshed\\33BusRadial.dss')
DSSSolution = DSS.Solution
DSSMon = DSS.Monitors

DSSSolution.Solve()
# Compile the circuit

# Get Incoming Flow from a line
LineInfo=getLineInfo(DSS,['Line_30'])
bus1power=[d['bus1powerreal'] for d in LineInfo]
print(bus1power)

#setLineInfo(DSSObj,['line_29'],'Enabled',[1],0)
DSS.CktElement.Name='Line_29'
DSS.CktElement.Enabled=0

LineInfo=getLineInfo(DSS,['Line_30'])
bus1power=[d['bus1powerreal'] for d in LineInfo]
print(bus1power)

# Get Values for two specific loads, the getloadinfo returns a list of dictionary objects
# # LoadInfo=getLoadInfo(DSSObj,['load_645','load_611'])

 # Getting the real power of the loads
# # LoadRealPower=[d['kw'] for d in LoadInfo]
# List does not support arithmatic, so you can convert it to an array
# # LoadRealPowerArray=np.asarray(LoadRealPower)


 # Get Values for all loads
# # value=getLoadInfo(DSSObj,[])
# # LoadRealPower=[d['kw'] for d in value]
# List does not support arithmatic, so you can convert it to an array
# # LoadRealPowerArray=np.asarray(LoadRealPower)
# # LoadNames=[d['name'] for d in value]

# Change the load power by a factor of 2 and then rerun a simulation
# # LoadRealPowerArray= 2 * LoadRealPowerArray
# # setLoadInfo(DSSObj,LoadNames,'kw',LoadRealPowerArray.tolist(),0)
# # dssSolution.Solve()
# print(value)



# Get some information about the buses 
# # BusInfo=getBusInfo(DSSObj,['bus_634','bus_645'])
# # print(BusInfo)


result= DSSStartup()
dssText=result['dsstext']
dssSolution=result['dsssolution']
dssCircuit=result['dsscircuit']
DSSObj=result['dssobj']
# Compile the circuit
dssText.Command="Compile C:/feeders/feeder33_Meshed/33BusRadial.dss"
# Run a Power Flow Solution
dssSolution.Solve()

DSSObj.ActiveCircuit.SetActiveElement('line_29')
DSSObj.ActiveCircuit.ActiveElement.Enabled