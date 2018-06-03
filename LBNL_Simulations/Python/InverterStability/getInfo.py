import numpy as np

def getLoadInfo(DSSObj,loadname):
    # TotalLoadProperty=6
    DSSCircuit = DSSObj.ActiveCircuit
    Loads = DSSCircuit.Loads
    if (len(loadname)==0):
        loadname= DSSCircuit.Loads.AllNames
    # print(loadname)
    # This further checking makes sure the model has loads
        if (len(loadname)==0):
            print('The Compiled Model Does not Have any Load.')
            return 0
    # load=np.zeros(shape=(len(loadname),TotalLoadProperty))
    # load=np.empty(shape=(len(loadname),),dtype=object)
    # LoadList=np.empty(shape=(len(loadname),),dtype=object)
    LoadList=[]
    for row in range(len(loadname)):
        loaddict = {}
        Loads.name = loadname[row]
        loaddict['kw']=Loads.kW
        loaddict['kvar']=Loads.kvar
        loaddict['kva'] = Loads.kva
        loaddict['kv'] = Loads.kV
        loaddict['pf'] = Loads.PF
        loaddict['model'] = Loads.model
        loaddict['name'] = Loads.name
        LoadList.append(loaddict)
        # LoadList[row]=loaddict
        # ColumnHeaders = ('kw','kvar','kva','kv','pf','model')
    # return pd.DataFrame(load,columns=ColumnHeaders,index=tuple(loadname))
    return LoadList


def getLineInfo(DSSObj,linename):
    DSSCircuit = DSSObj.ActiveCircuit
    Lines = DSSCircuit.Lines
    if (len(linename)==0):
        linename=DSSCircuit.Lines.AllNames
        if (len(linename)==0):
            print('The Compiled Model Does not Have any Line.')
            return 0
    LineList=[]
    for row in range(len(linename)):
        linedict={}
        Lines.name=linename[row]
        linedict['r1']=Lines.R1
        linedict['name']=linename[row]
        linedict['x1']=Lines.X1
        linedict['length']=Lines.length
        DSSCircuit.SetActiveElement('line.'+linename[row])
        lineBusNames = DSSCircuit.ActiveElement.BusNames
        linedict['bus1'] = lineBusNames[0]
        linedict['bus2'] = lineBusNames[1]
        linedict['enabled']=DSSCircuit.ActiveElement.Enabled
        if (not DSSCircuit.ActiveElement.Enabled):
            continue
        power = np.asarray(DSSCircuit.ActiveCktElement.Powers)
        bus1power = power[0: (int)(len(power) / 2)]
        bus2power = power[(int)(len(power) / 2):]
        bus1powerreal=bus1power[0::2]
        bus1powerrective = bus1power[1::2]
        bus2powerreal = bus2power[0::2]
        bus2powerrective = bus2power[1::2]
        numofphases = DSSCircuit.ActiveElement.NumPhases
        linedict['numofphases'] = numofphases
        phaseinfo = np.asarray(['.1' in lineBusNames[0], '.2' in lineBusNames[0], '.3' in lineBusNames[0]])
        if (numofphases==3):
            bus1PhasePowerReal=bus1powerreal
            bus2PhasePowerReal = bus2powerreal
            bus1PhasePowerReactive=bus1powerrective
            bus2PhasePowerReactive = bus2powerrective
        elif (numofphases==1):
            bus1PhasePowerReal=bus1powerreal[0]*phaseinfo
            bus2PhasePowerReal = bus2powerreal[0] * phaseinfo
            bus1PhasePowerReactive = bus1powerrective[0] * phaseinfo
            bus2PhasePowerReactive = bus2powerrective[0] * phaseinfo
        elif numofphases==2:
            A = np.zeros(shape=(3, 2))
            col = -1
            for i in range(len(phaseinfo)):
                if (phaseinfo[i]):
                    col += 1
                    A[i][col] = 1
            bus1PhasePowerReal=np.dot(A,bus1powerreal)
            bus2PhasePowerReal = np.dot(A, bus2powerreal)
            bus1PhasePowerReactive = np.dot(A, bus1powerrective)
            bus2PhasePowerReactive = np.dot(A, bus2powerrective)
        else:
            continue
        linedict['bus1phasepowerreal']=bus1PhasePowerReal
        linedict['bus2phasepowerreal']=bus2PhasePowerReal
        linedict['bus1phasepowerreactive']=bus1PhasePowerReactive
        linedict['bus2phasepowerreactive'] = bus2PhasePowerReactive
        linedict['bus1powerreal']=np.sum(bus1PhasePowerReal)
        linedict['bus2powerreal'] = np.sum(bus2PhasePowerReal)
        linedict['bus1powerreactive'] = np.sum(bus1PhasePowerReactive)
        linedict['bus2powerreactive'] = np.sum(bus1PhasePowerReactive)
        LineList.append(linedict)
    return LineList

def getBusInfo(DSSObj,busname):
    DSSCircuit = DSSObj.ActiveCircuit
    if (len(busname)==0):
        busname=DSSCircuit.DSSCircuit.AllBusNames
    BusList=[]
    for row in range(len(busname)):
        busdict={}
        busdict['name']=busname[row]
        DSSCircuit.SetActiveBus(busname[row])
        noofphases=DSSCircuit.ActiveBus.NumNodes
        busdict['nodes']=noofphases
        complexpuvoltage=np.reshape(np.asarray(DSSCircuit.ActiveBus.puVoltages),newshape=(noofphases,2))
        phasevoltagespu=[]
        for i in range(noofphases):
            phasevoltagespu.append(abs(complex(complexpuvoltage[i][0],complexpuvoltage[i][1])))
        phasevoltagespu=np.asarray(phasevoltagespu)
        busdict['phasevoltagepu'] = phasevoltagespu
        busdict['voltagepu']=np.mean(phasevoltagespu)
        BusList.append(busdict)
    return BusList