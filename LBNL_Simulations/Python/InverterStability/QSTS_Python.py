# This code actually shows to run a QSTS analysis and do custom run
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
DSSText.command = 'Compile C:/feeders/feeder13_U_R_Pecan/simulation.dss'
DSSText.command ='Solve mode="daily" number=1440 stepsize="1s"'

# regulators=DSSCircuit.RegControls
# meter name from which I will retrieve the data
DSSMon.Name='meter_634'
# just printing the meter headers
print(DSSMon.header)
# Unfortunately the channel does not have the time information (still under investigation)
time=np.arange(1,1441,1)
# Reading the Voltage
Voltage_Phasea=np.asarray(DSSMon.Channel(1))
Voltage_Phaseb=np.asarray(DSSMon.Channel(3))
Voltage_Phasec=np.asarray(DSSMon.Channel(5))
Voltage=Voltage_Phasea+Voltage_Phaseb+Voltage_Phasec
# dividing by the base and also taking average
plt.plot(time,Voltage/(3*2400))
plt.show()
# Show how to use the getreginfo function
regs=getRegInfo(DSSObj,[])
print(regs)
