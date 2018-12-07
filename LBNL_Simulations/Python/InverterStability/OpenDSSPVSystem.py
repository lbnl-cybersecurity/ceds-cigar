# This code is an exmaple to show the PVsystem capability of OpenDSS through Python
# Also demostrates how to retrieve the monitor data
# The code also shows an exmaple of manupulating the curve

from DSSStartup import DSSStartup
from setInfo import *
from getInfo import *
import matplotlib.pyplot as plt
import numpy as np

result= DSSStartup()
DSSText=result['dsstext']
DSSSolution=result['dsssolution']
DSSCircuit=result['dsscircuit']
DSSObj=result['dssobj']

DSSText.command = 'Compile C:/feeders/feeder13_U_R_Pecan_Solar/feeder13_U_R_Pecan_Solar.dss'
# This is a quick way to change properties of a specific group of device
setRegInfo(DSSObj,list(DSSCircuit.RegControls.AllNames),'enabled',[0]* len(DSSCircuit.RegControls.AllNames))
setCapControlInfo(DSSObj,list(DSSCircuit.CapControls.AllNames),'enabled',[0]*len(DSSCircuit.CapControls.AllNames))


DSSSolution.Mode=1 # 1 represents daily mode, 2 represents yearly mode
DSSSolution.Number=1440 # Solutions Per Solve Command
DSSSolution.StepSize=1 # Stepsize= 1s
DSSText.Command='Set ControlMode=Time' # Refer to OpenDSS documentation
DSSSolution.Solve()


DSSMon=DSSCircuit.Monitors
DSSMon.Name='Meter_632_power'
print('Monitor Name: ' + DSSMon.Name )

time=3600*np.asarray((DSSMon.dblHour))
Real_power=np.asarray(DSSMon.Channel(1))+np.asarray(DSSMon.Channel(3))+np.asarray(DSSMon.Channel(5))
Reactive_power=np.asarray(DSSMon.Channel(2))+np.asarray(DSSMon.Channel(4))+np.asarray(DSSMon.Channel(6))

# Generation is negative in OpenDSS hence, multiply by -1
plt.figure()
plt.plot(time,-Real_power)
plt.title('Real Power from Solar at Bus 632')
plt.show()

plt.figure()
plt.plot(time,-Reactive_power)
plt.title('Reactive Power from Solar at Bus 632')
plt.show()

DSSMon=DSSCircuit.Monitors
DSSMon.Name='Meter_692_power'

Real_power=np.asarray(DSSMon.Channel(1))+np.asarray(DSSMon.Channel(3))+np.asarray(DSSMon.Channel(5))
Reactive_power=np.asarray(DSSMon.Channel(2))+np.asarray(DSSMon.Channel(4))+np.asarray(DSSMon.Channel(6))

plt.figure()
plt.plot(time,-Real_power)
plt.title('Real Power from Solar at Bus 692')
plt.show()

plt.figure()
plt.plot(time,-Reactive_power)
plt.title('Reactive Power from Solar at Bus 692')
plt.show()


# Getting a specific xycurve
xycurves=getXYCurveInfo(DSSObj,['vw_curve'])
# because it returns a list
vw_curve=xycurves[0]
# print to console
print(vw_curve)
# Change one parameter of the curve
vw_curve['xarray']=np.asarray([1,1,1,1])
# making sure that the change happened
print(vw_curve)
# set the curveinformation
setXYCurveInfo(DSSObj,['vw_curve'],[vw_curve])

# make sure that the change happened
xycurves=getXYCurveInfo(DSSObj,['vw_curve'])
vw_curve=xycurves[0]
print(vw_curve)

# Error Checking
vw_curve['xarray']=np.asarray([1,1,1,1,1])
setXYCurveInfo(DSSObj,['vw_curve'],[vw_curve])


