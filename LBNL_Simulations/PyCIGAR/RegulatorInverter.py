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
DSSText.Command="Compile C:/feeders/34BusLTC/Radial34Bus.dss"

#  Enabling the controls for both regulators and Inverters; OpenDSS only allow to change the PVSystem Parameters, not the INvControl parameters
DSSText.command='BatchEdit PVSystem..* pctpmpp=100'
DSSText.command='BatchEdit InvControl..* enabled=Yes'
DSSText.command='BatchEdit RegControl..* enabled= Yes'

#  Extracting the regulator Info
regInfo=getRegInfo(DSSObj,[])
MaxTap=[d['maxtapchange'] for d in regInfo]
print('Max Taps for Regulators:'+ str(MaxTap))
#  Set the regulator Info: check OpenDSS documentation for the details 
setRegInfo(DSSObj,['ltc-t_02'],'maxtapchange',[1],0)


#  Set the solution paramters; Stepsize should always be in seconds
setSolutionParams(DSSObj,'daily',1440,60,'time',10000,1000)
# Run a Power Flow Solution
DSSSolution.Solve()


# Grab the monitor data 
DSSMon=DSSCircuit.Monitors
DSSMon.Name='tapMonitor'
#  print the monitor name
print('Monitor Name:' + str(DSSMon.name))
time=3600*np.asarray((DSSMon.dblHour))

tap=np.asarray(DSSMon.Channel(1))

DSSMon.Name='solar 01'
#  print the monitor name
print('Monitor Name:' + str(DSSMon.name))
Real_power=np.asarray(DSSMon.Channel(1))+np.asarray(DSSMon.Channel(3))+np.asarray(DSSMon.Channel(5))
Reactive_power=np.asarray(DSSMon.Channel(2))+np.asarray(DSSMon.Channel(4))+np.asarray(DSSMon.Channel(6))

BaseVoltage=24.9*1000/(3**0.5)
#  print the monitor name
DSSMon.Name='solar 01 VI'
Voltage=np.asarray(DSSMon.Channel(1))+np.asarray(DSSMon.Channel(3))+np.asarray(DSSMon.Channel(5))


#  The section for plotting data
plt.figure()
plt.plot(time,tap,label='Transformer Tap')
plt.title('Transformer Tap Position')
plt.legend()
plt.show()


plt.figure()
plt.plot(time,-Real_power,label='Real Power')
plt.plot(time,-Reactive_power,label='Reactive Power')
plt.title('Real and Reactive Power from Solar ')
plt.legend()
plt.show()

plt.figure()
plt.plot(time,Voltage/3/BaseVoltage,label='Voltage')
plt.title('Average Per Unit Bus Voltage')
plt.legend()
plt.show()

