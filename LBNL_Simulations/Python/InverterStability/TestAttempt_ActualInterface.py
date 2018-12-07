from DSSStartup import DSSStartup
from setInfo import *
from getInfo import *

result= DSSStartup()
dssText=result['dsstext']
dssSolution=result['dsssolution']
dssCircuit=result['dsscircuit']
DSSObj=result['dssobj']

dssText.Command="Compile C:/feeders/feeder13_B_R/feeder13BR.dss "
dssSolution.Solve()

loadName = "Load.load_645"
dssCircuit.Loads.Name = loadName.split(".")[1]
oldkW = dssCircuit.Loads.kW
dssCircuit.Loads.kW = 200
dssSolution.Solve()
dssCircuit.SetActiveElement('Line.L_U_650')
print(dssCircuit.ActiveElement.Powers)

dssText.Command = 'Load.load_645.kW=400'
print(dssCircuit.Loads.kW)
dssSolution.Solve()
dssCircuit.SetActiveElement('Line.L_U_650')
print(dssCircuit.ActiveElement.Powers)

dssText.Command = 'Load.load_645.kW=300'
print(dssCircuit.Loads.kW)
dssSolution.Solve()
dssCircuit.SetActiveElement('Line.L_U_650')
print(dssCircuit.ActiveElement.Powers)

setLoadInfo(DSSObj,['load_645','load_601'],'kw',[10,120],1)
loadName = "Load.load_645"
dssCircuit.Loads.Name = loadName.split(".")[1]
oldkW = dssCircuit.Loads.kW
print(oldkW)

loadName = "Load.load_611"
dssCircuit.Loads.Name = loadName.split(".")[1]
oldkW = dssCircuit.Loads.kW
print(oldkW)

setSourceInfo(DSSObj,['source'],'pu',[1.02],0)
dssSolution.Solve()
dssCircuit.SetActiveElement('Line.L_U_650')
print(dssCircuit.ActiveElement.Powers)

x=getLoadInfo(DSSObj,['load_645'])
y=[d['voltagePU'] for d in x]
print(y)