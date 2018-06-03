# This code runs the 13 Node Balanced Model using the Python OpenDSS interface
# The code also runs the custom FBS functions to compare the results
# Please make sure you have the necessary libraries, and the required libraries are in the same folder with the MATLAB code
# Be advised: While Running OpenDSS, python changes the current directory, hence the matlab code is run first to avoid additional coding
# Make sure you have the updated version of Anaconda for all libraries to run, OpenDSS installed 
# For a custom made dictionary, the key is always in lower case
from DSSStartup import DSSStartup
from setInfo import *
from getInfo import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import tan,acos


#  Initiating the global parameters
SlackBusVoltage=1.0
power_factor=0.9
reactivepowercontribution=tan(acos(power_factor)) # Q = P tan(Theta)
IncludeSolar=1
# This Following Block allows to run the matlab code from python
import matlab.engine as matlab
def start_matlab():
    return matlab.start_matlab()
def quit_matlab(matlab_engine):
    matlab_engine.quit()
def CustomFBS(matlab_engine,NodeVoltageToPlot):
    return matlab_engine.FBS(NodeVoltageToPlot,IncludeSolar,nargout=2) # As the function will return two outputs, nargout=2

# The values are to converted to array as the matlab function retunrs a tuple

NodeVoltageToPlot=634
matlab_engine=start_matlab()
VoltageFBS,SubstationRealPowerFBS=CustomFBS(matlab_engine,NodeVoltageToPlot)
quit_matlab(matlab_engine)

VoltageFBS=np.asarray(VoltageFBS)
SubstationRealPowerFBS=np.asarray(SubstationRealPowerFBS)
##############################################

#  DSSStartup() returns a dictionary
result= DSSStartup()
DSSText=result['dsstext']
DSSSolution=result['dsssolution']
DSSCircuit=result['dsscircuit']
DSSObj=result['dssobj']
current_directory=os.getcwd()
# Compile the circuit
DSSText.Command="Compile C:/feeders/feeder13_B_R/feeder13BR.dss "
# Run a Power Flow Solution
DSSSolution.Solve()

#  The Load bus names and bus list are derived from MATLAB to match the plots
LoadBusNames=['load_634','load_645','load_646','load_652','load_671','load_675','load_692','load_611']
LoadList=np.array([6,7,8,13,3,12,11,10])-1
#  The solar value and load profile are custom generated from the MATLAB code directly, the values are divided by 1000 to convert to kw
Load=pd.read_csv('C:/feeders/13busload.csv',header=None)
Load=1/1000*(Load.values)
Solar=pd.read_csv('C:/feeders/Solar.csv',header=None)
Solar=1/1000*(Solar.values)
TotalTimeSteps,Nodes= Load.shape # Initialize the total time steps according to the data

VoltageOpenDSS=np.zeros(shape=(TotalTimeSteps,))
SubstationRealPowerOpenDSS=np.zeros(shape=(TotalTimeSteps,))
setSourceInfo(DSSObj,['source'],'pu',[SlackBusVoltage]) # Setting the slack bus voltage
BusVoltageToPolt='bus_'+ str(NodeVoltageToPlot)
for ksim in range(TotalTimeSteps):
	# Setting the real and reactive power of the loads
    setLoadInfo(DSSObj,LoadBusNames,'kw',Load[ksim][LoadList]-IncludeSolar*Solar[ksim][LoadList])
    setLoadInfo(DSSObj, LoadBusNames, 'kvar', reactivepowercontribution*(Load[ksim][LoadList] - IncludeSolar*Solar[ksim][LoadList]))
    # Solving the OpenDSS Power FLow
	DSSSolution.Solve()
	#
    LineInfo=getLineInfo(DSSObj,['L_U_650'])
    bus1power = [d['bus1powerreal'] for d in LineInfo]
    SubstationRealPowerOpenDSS[ksim]=bus1power[0] # This is done as the variable is a list, and the first element of the list, this can be done by doing a list.append, but array is done for speed issue
    BusInfo=getBusInfo(DSSObj,[BusVoltageToPolt])
    voltagepu=[d['voltagepu'] for d in BusInfo]
    VoltageOpenDSS[ksim]=voltagepu[0]

## Plot Functions
time=np.arange(0,TotalTimeSteps,1)
plt.plot(time,VoltageOpenDSS,time,VoltageFBS[0,:])
plt.legend(['OpenDSS','FBS'])
plt.ylabel('Voltage (pu)')
plt.title('VOltage at Node ' + str(NodeVoltageToPlot))
plt.show()
plt.figure()
plt.plot(time,SubstationRealPowerOpenDSS,time,SubstationRealPowerFBS[0,:])
plt.legend(['OpenDSS','FBS'])
plt.ylabel('Real Power (kW)')
plt.title('Real Power From Substation')
plt.show()

