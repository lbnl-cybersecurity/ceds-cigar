# This code actually shows to run a QSTS analysis and do custom run illustrating the capability
# of the library to do regulator control and capacitor control
from DSSStartup import DSSStartup
from setInfo import *
from getInfo import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



result= DSSStartup()
DSSText=result['dsstext']
DSSSolution=result['dsssolution']
DSSCircuit=result['dsscircuit']
DSSObj=result['dssobj']
DSSMon=DSSCircuit.Monitors
DSSText.command = 'Compile C:/feeders/feeder13_U_R_Pecan/feeder13_U_R_Pecan.dss'
# showing ow to use the designed set functions
setRegInfo(DSSObj,['lvr-lvr_01'],'maxtapchange',[16])
setCapInfo(DSSObj,['cap_675'],'kvar',[600])
DSSSolution.Mode=1 # 1 represents daily mode
DSSSolution.Number=1440 # Solutions Per Solve Command
DSSSolution.StepSize=1 # Stepsize= 1s
DSSSolution.Solve()


# Show how to use the getreginfo function
regs=getRegInfo(DSSObj,[])
#print(regs)
MaxTapChangeValues=[d['maxtapchange'] for d in regs]
print('Maximum Tap Change Values for the Regulators: ' + str(MaxTapChangeValues) + '\n')

capcontrols=getCapControlInfo(DSSObj,[])
PTRatio= [d['ptratio'] for d in capcontrols]
print('PTratio Values for the Capcontrols: ' + str(PTRatio) + '\n' )


# regulators=DSSCircuit.RegControls
# meter name from which I will retrieve the data
DSSMon.Name='meter_634'
# just printing the meter headers
print(DSSMon.header)
# Unfortunately the channel does not have the time information (still under investigation)
time=3600*np.asarray(list(DSSMon.dblHour)) # Multiplying by 3600 converts it to seconds
# Reading the Voltage
Voltage_Phasea=np.asarray(DSSMon.Channel(1))
Voltage_Phaseb=np.asarray(DSSMon.Channel(3))
Voltage_Phasec=np.asarray(DSSMon.Channel(5))
Voltage=Voltage_Phasea+Voltage_Phaseb+Voltage_Phasec
# dividing by the base and also taking average
plt.plot(time,Voltage/(3*2400))
plt.show()


DSSMon.Name='VAR_Meter_675'
CapQ=-1* (np.asarray(DSSMon.Channel(2)) + np.asarray(DSSMon.Channel(4)) + np.asarray(DSSMon.Channel(6))) # Because generation is negative from OpenDSS perspective
plt.figure()
plt.plot(time,CapQ)
plt.show()



# print(getCapControlInfo(DSSObj,[]))