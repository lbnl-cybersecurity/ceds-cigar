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
import os

result= DSSStartup()
DSSText=result['dsstext']
DSSSolution=result['dsssolution']
DSSCircuit=result['dsscircuit']
DSSObj=result['dssobj']
NodeVoltageToPlot=634
IncludeSolar=1
SlackBusVoltage=1.0
DSSText.Command="Compile C:/feeders/feeder13_U_R_Pecan_Solar/feeder13_U_R_Pecan_Solar.dss"

LoadBusNames=['load_634','load_645','load_646','load_652','load_671','load_675','load_692','load_611']
LoadList=np.array([6,7,8,13,3,12,11,10])-1
#get the real load from Texas day
Load=pd.read_csv('C:/feeders/13busload.csv',header=None)
Load=1/1000*(Load.values)
#get the real solar from Texas day
Solar=pd.read_csv('C:/feeders/Solar.csv',header=None)
Solar=1/1000*(Solar.values)
TotalTimeSteps,Nodes= Load.shape # Initialize the total time steps according to the data

DSSSolution.Mode=1 # 1 represents daily mode, 2 represents yearly mode
DSSSolution.Number=TotalTimeSteps # Solutions Per Solve Command
DSSSolution.StepSize=1 # Stepsize= 1s
DSSSolution.MaxControlIterations=1000 #Increase the number of maximum control iterations to make sure the system can solve the power flow
DSSSolution.MaxIterations=100 #Increasing the number of power flow iterations to achieve convergence

DSSText.Command='Set ControlMode=Time' # Refer to OpenDSS documentation

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
    LineInfo=getLineInfo(DSSObj,['L_U_650'])
    bus1power = [d['bus1powerreal'] for d in LineInfo]
    SubstationRealPowerOpenDSS[ksim]=bus1power[0] # This is done as the variable is a list, and the first element of the list, this can be done by doing a list.append, but array is done for speed issue
    BusInfo=getBusInfo(DSSObj,[BusVoltageToPolt])
    voltagepu=[d['voltagepu'] for d in BusInfo]
    VoltageOpenDSS[ksim]=voltagepu[0]
