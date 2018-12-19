import numpy as np

def getLoadInfo(DSSObj,loadname):
    Loads = DSSObj.Loads
    if (len(loadname)==0):
        loadname = Loads.AllNames()
    # This further checking makes sure the model has loads
        if (len(loadname)==0):
            print('The Compiled Model Does not Have any Load.')
            return 0
    LoadList=[]
    for row in range(len(loadname)):
        loaddict = {}
        Loads.Name(loadname[row])
        loaddict['kw']= Loads.kW()
        loaddict['kvar']= Loads.kvar()
        loaddict['kva'] = Loads.kVABase()
        loaddict['kv'] = Loads.kV()
        loaddict['pf'] = Loads.PF()
        loaddict['model'] = Loads.Model()
        loaddict['name'] = Loads.Name()        
        DSSObj.Circuit.SetActiveElement('load.'+loadname[row])

        voltage=np.asarray(DSSObj.CktElement.VoltagesMagAng())
        # Multiplying by 1000 to convert it to Voltage, the square root 3 part can be made more robust in future but currently it is assumed to be balanced model
        loaddict['voltagePU'] = (voltage[0]+voltage[2]+voltage[4])/(DSSObj.CktElement.NumPhases()*(Loads.kV()*1000/(3**0.5)))
        LoadList.append(loaddict)
    return LoadList

def getCapInfo(DSSObj,capname):
    
    Capacitors=DSSObj.Capacitors
    if (len(capname)==0):
        capname=Capacitors.AllNames()
        if (len(capname)==0):
            print('The Complied model does not have any capacitors.')
            return 0
    CapList=[]
    for row in range(len(capname)):
        capdict = {}
        Capacitors.Name(capname[row])
        capdict['kvar']=Capacitors.kvar()
        capdict['kv'] = Capacitors.kv()
        capdict['numsteps'] = Capacitors.NumSteps()
        capdict['states']=Capacitors.States()
        capdict['name'] = Capacitors.Name()
        CapList.append(capdict)
    return CapList


def getCapControlInfo (DSSObj,capcontrolname):
#    DSSCircuit=DSSObj.ActiveCircuit
    CapControls=DSSObj.CapControls
    if (len(capcontrolname)==0):
        capcontrolname=CapControls.AllNames()
        if (len(capcontrolname)==0):
            print('The Compiled Model does not have any capacitor control objects.')
            return 0
    CapControlList=[]
    for row in range(len(capcontrolname)):
        capcontroldict={}
        CapControls.Name(capcontrolname[row])
        capcontroldict['ptratio']=CapControls.PTratio()
        capcontroldict['ctratio'] = CapControls.CTratio()
        capcontroldict['onsetting'] = CapControls.ONSetting()
        capcontroldict['offsetting'] = CapControls.OFFSetting()
        capcontroldict['delay'] = CapControls.Delay()
        capcontroldict['delayoff'] = CapControls.DelayOff()
        capcontroldict['deadtime'] = CapControls.DeadTime()
        capcontroldict['vmax'] = CapControls.Vmax()
        capcontroldict['vmin'] = CapControls.Vmin()
        capcontroldict['capacitor'] = CapControls.Capacitor()
        capcontroldict['monitoredobj'] = CapControls.MonitoredObj()
        capcontroldict['monitoredterm'] = CapControls.MonitoredTerm()
        capcontroldict['name'] = CapControls.Name()
        DSSObj.CktElement.Name='CapControl.'+CapControls.Name
        capcontroldict['enabled']=DSSObj.CktElement.Enabled
        CapControlList.append(capcontroldict)
    return CapControlList


def getRegInfo(DSSObj,regname):
#    DSSCircuit = DSSObj.ActiveCircuit
    regcontrols = DSSObj.RegControls
    if (len(regname)==0):
        regname= regcontrols.AllNames()
        if (len(regname)==0):
            print('The Compiled Model Does not Have any Regulators.')
            return 0
    RegList=[]
    for row in range(len(regname)):
        regdict = {}
        regcontrols.name = regname[row]
        regdict['maxtapchange']=regcontrols.MaxTapChange
        regdict['delay']=regcontrols.Delay
        regdict['tapdelay'] = regcontrols.TapDelay
        regdict['tapnumber'] = regcontrols.TapNumber
        regdict['transformer'] = regcontrols.Transformer
        regdict['name'] = regcontrols.name
        DSSObj.CktElement.Name='RegControl.' + regcontrols.Name
        regdict['enabled'] = DSSObj.CktElement.Enabled
        RegList.append(regdict)
    return RegList


def getLineInfo(DSSObj,linename):
#    DSSCircuit = DSSObj.ActiveCircuit
    Lines = DSSObj.Lines
    if (len(linename)==0):
        linename=Lines.AllNames()
        if (len(linename)==0):
            print('The Compiled Model Does not Have any Line.')
            return 0
    LineList=[]
    for row in range(len(linename)):
        linedict={}
        Lines.Name(linename[row])
        linedict['r1']=Lines.R1()
        linedict['name']=linename[row]
        linedict['x1']=Lines.X1()
        linedict['length']=Lines.Length()
        DSSObj.CktElement.Name='line.'+linename[row]
        lineBusNames = DSSObj.CktElement.BusNames()
        linedict['bus1'] = lineBusNames[0]
        linedict['bus2'] = lineBusNames[1]
        linedict['enabled']=DSSObj.CktElement.Enabled
#        if (not DSSObj.CktElement.Enabled):
#            continue
        power = np.asarray(DSSObj.CktElement.Powers())
        bus1power = power[0: (int)(len(power) / 2)]
        bus2power = power[(int)(len(power) / 2):]
        bus1powerreal=bus1power[0::2]
        bus1powerrective = bus1power[1::2]
        bus2powerreal = bus2power[0::2]
        bus2powerrective = bus2power[1::2]
        numofphases = DSSObj.CktElement.NumPhases()
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
        busname=DSSCircuit.AllBusNames()
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

def getXYCurveInfo(DSSObj,xycurvename):
#    DSSCircuit = DSSObj.ActiveCircuit
    XYCurves = DSSObj.XYCurves
    XYCurveList=[]
    if (XYCurves.Count==0):
        return XYCurveList
    if (len(xycurvename)==0):
        XYCurves.First
        for i in range(XYCurves.Count):
            xycurvename.append(XYCurves.Name)
            XYCurves.Next
    for row in range (len(xycurvename)):
        xycurvedict={}
        XYCurves.Name=xycurvename[row]
        xycurvedict['name']= xycurvename[row]
        xycurvedict['npts'] =XYCurves.Npts
        xycurvedict['xarray'] = np.asarray(XYCurves.Xarray)
        xycurvedict['yarray'] = np.asarray(XYCurves.Yarray)
        XYCurveList.append(xycurvedict)
    return XYCurveList