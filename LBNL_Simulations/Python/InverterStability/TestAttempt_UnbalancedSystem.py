# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:50:19 2018

@author: shamm
"""

import win32com.client
import opendssdirect as dss
from opendssdirect.utils import run_command
import numpy as np
from getInfo import *

dssObj = win32com.client.Dispatch("OpenDSSEngine.DSS")

DSSText = dssObj.Text
DSSCircuit = dssObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution
DSSElem = DSSCircuit.ActiveCktElement
DSSBus = DSSCircuit.ActiveBus
import os
print(os.getcwd())
DSSText.Command="compile C:/feeders/TwoPhaseLoad/TwoPhaseLoad.dss "
DSSSolution.Solve()
DSSCircuit.SetActiveElement('Line.conn_03')
print(DSSCircuit.ActiveElement.Powers)
value=getLineInfo(dssObj,['conn_03'])
lineBusNames = DSSCircuit.ActiveElement.BusNames
x='busbar 01.2.3'
phaseinfo=np.asarray(['.1' in x,'.2' in x,'.3' in x])
A=np.zeros(shape=(3,2))
col=-1
for i in range(len(phaseinfo)):
    if (phaseinfo[i]):
        col +=1
        A[i][col]=1

print(value)
#load=np.loadtxt('C:/feeders/13busload.csv')
import pandas as pd
value=pd.read_csv('C:/feeders/13busload.csv')




 