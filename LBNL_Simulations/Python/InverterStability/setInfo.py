def setLoadInfo(DSSObj,loadname,property,value,NameChecker=0):
    try:
        property=property.lower()
    except:
        print('String Expected')
        return DSSObj,False

    DSSCircuit = DSSObj.ActiveCircuit
    Loads = DSSCircuit.Loads

    if (NameChecker !=0):
        AllLoadNames = Loads.AllNames
        match_values=0
        for i in range(len(loadname)):
            if any(loadname[i] in item for item in AllLoadNames):
                match_values+=1
        if match_values != len(loadname):
            print('Load Not Found')
            return DSSObj,False
    if (len(loadname) != len(value)):
        return DSSObj,False
    for counter in range(len(value)):
        Loads.name=loadname[counter]
        if (property=='kw'):
            Loads.kW = value[counter]
        elif (property=='kvar'):
            Loads.kvar = value[counter]
        elif (property=='kva'):
            Loads.kva = value[counter]
        elif (property=='pf'):
            Loads.PF=value[counter]
        else:
            print('Property Not Found')
            return DSSObj,False
    return DSSObj,True


def setCapInfo(DSSObj,capname,property,value,NameChecker=0):
    try:
        property = property.lower()
    except:
        print('String Expected')
        return DSSObj, False
    DSSCircuit = DSSObj.ActiveCircuit
    Caps=DSSCircuit.Capacitors

    if (NameChecker !=0):
        AllCapNames = Caps.AllNames
        match_values = 0
        for i in range(len(capname)):
            if any(capname[i] in item for item in AllCapNames):
                match_values+=1
            if match_values != len(capname):
                print('One of the Caps Not Found.')
                return DSSObj, False
    if (len(capname) != len(value)):
        return DSSObj,False
    for counter in range(len(value)):
        Caps.name=capname[counter]
        if (property=='kvar'):
            Caps.kvar = value[counter]
        elif (property=='kv'):
            Caps.kv = value[counter]
        elif (property=='numsteps'):
            Caps.NumSteps=value[counter]
        elif (property=='states'):
            Caps.states = [int(d) for d in str(value[counter])] # This actually separate the digits
        else:
            print('Property Not Found')
            return DSSObj, False
    return DSSObj, True

def setCapControlInfo(DSSObj,capcontrolname,property,value,NameChecker=0):
    try:
        property = property.lower()
    except:
        print('String Expected')
        return DSSObj, False
    DSSCircuit = DSSObj.ActiveCircuit
    CapControls = DSSCircuit.CapControls
    if (NameChecker !=0):
        AllCapControlNames = CapControls.AllNames
        match_values = 0
        for i in range(len(capcontrolname)):
            if any(capcontrolname[i] in item for item in AllCapControlNames):
                match_values+=1
            if match_values != len(capcontrolname):
                print('One of the Capcontrols Not Found.')
                return DSSObj, False
    if (len(capcontrolname) != len(value)):
        return DSSObj,False
    for counter in range(len(value)):
        CapControls.name=capcontrolname[counter]
        if (property=='delay'):
            CapControls.Delay = value[counter]
        elif (property=='delayoff'):
            CapControls.DelayOFF = value[counter]
        elif (property == 'offsetting'):
            CapControls.OFFsetting=value[counter]
        elif (property=='onsetting'):
            CapControls.ONsetting=value[counter]
        elif (property=='element'):
            CapControls.element=value[counter]
        elif (property=='terminal'):
            CapControls.terminal=value[counter]
        elif (property=='vmax'):
            CapControls.Vmax=value[counter]
        elif (property == 'vmin'):
                CapControls.Vmin = value[counter]
        elif (property == 'ptratio'):
            CapControls.PTratio = value[counter]
        elif (property == 'ctratio'):
            CapControls.CTratio = value[counter]
        elif (property =='enabled'):
            DSSCircuit.SetActiveElement('CapControl.' + CapControls.name)
            DSSCircuit.ActiveElement.Enabled = value[counter]
        else:
            print('Property Not Found')
            return DSSObj, False
    return DSSObj, True

def setSourceInfo(DSSObj,sourcename,property,value,NameChecker=0):
    try:
        property=property.lower()
    except:
        print('String Expected')
        return DSSObj,False

    DSSCircuit = DSSObj.ActiveCircuit
    Sources = DSSCircuit.Vsources

    if (NameChecker !=0):
        AllSourceNames = Sources.AllNames
        match_values=0
        for i in range(len(sourcename)):
            if any(sourcename[i] in item for item in AllSourceNames):
                match_values+=1
        if match_values != len(sourcename):
            print('Source Not Found')
            return DSSObj,False
    if (len(sourcename) != len(value)):
        return DSSObj,False
    for counter in range(len(value)):
        Sources.name=sourcename[counter]
        if (property=='pu'):
            Sources.pu = value[counter]
        elif (property=='basekv'):
            Sources.BasekV = value[counter]
            Sources.PF=value[counter]
        else:
            print('Property Not Found')
            return DSSObj,False
    return DSSObj,True


def setRegInfo(DSSObj,regname,property,value,NameChecker=0):
    try:
        property=property.lower()
    except:
        print('String Expected')
        return DSSObj,False

    DSSCircuit = DSSObj.ActiveCircuit
    regcontrols = DSSCircuit.RegControls
    DSSText = DSSObj.Text
    if (NameChecker !=0):
        AllRegNames = regcontrols.AllNames
        match_values=0
        for i in range(len(regname)):
            if any(regname[i] in item for item in AllRegNames):
                match_values+=1
        if match_values != len(regname):
            print('Regulator Not Found')
            return DSSObj,False
    if (len(regname) != len(value)):
        return DSSObj,False
    for counter in range(len(value)):
        regcontrols.name=regname[counter]
        if (property=='maxtapchange'):
            regcontrols.MaxTapChange = value[counter]
        elif (property=='delay'):
            regcontrols.Delay = value[counter]
        elif (property=='tapdelay'):
            regcontrols.TapDelay = value[counter]
        elif (property=='tapnumber'):
            regcontrols.tapnumber=value[counter]
        elif (property=='transformer'):
            regcontrols.Transformer=value[counter]
        elif (property=='enabled'):
            DSSCircuit.SetActiveElement('RegControl.' + regcontrols.name)
            DSSCircuit.ActiveElement.Enabled=value[counter]
        elif (property=='debugtrace'):
            if (value[counter]>0):
                DSSText.command = 'RegControl.'+ regname[counter] + '.debugtrace=yes'
            else:
                DSSText.command = 'RegControl.' + regname[counter] + '.debugtrace=no'
        elif (property=='eventlog'):
            if (value[counter]>0):
                DSSText.command = 'RegControl.'+ regname[counter] + '.eventlog=yes'
            else:
                DSSText.command = 'RegControl.' + regname[counter] + '.eventlog=no'
        else:
            print('Property Not Found')
            return DSSObj,False
    return DSSObj,True

def setXYCurveInfo(DSSObj,xycurvename,value,NameChecker=0):
    DSSCircuit = DSSObj.ActiveCircuit
    XYCurves = DSSCircuit.XYCurves
    if (XYCurves.Count==0):
        print('No XYCurve Found in the OpenDSS Model.')
        return DSSObj, False
    if not(NameChecker ==0):
        AllXYCurveNames = []
        XYCurves.First
        for i in range(XYCurves.Count):
            AllXYCurveNames.append(XYCurves.Name)
            XYCurves.Next
        match_values=0
        for i in range(len(xycurvename)):
            if any(xycurvename[i] in item for item in AllXYCurveNames):
                match_values+=1
        if match_values != len(xycurvename):
            print('Load Not Found')
            return DSSObj,False
    if not (len(value)==len(xycurvename)):
        print('Number of XYCurvenames and Number of property do not match.')
        return DSSObj, False
    for counter in range(len(xycurvename)):
        if (len(value[counter])!=4):
            print('You Need to Set All the properties.')
            return DSSObj, False
        xycurve=value[counter]
        if (value[counter]['npts'] != len(value[counter]['xarray']) or  value[counter]['npts'] != len(value[counter]['yarray'])):
            print('Input Value Mismatch')
            return DSSObj,False
        XYCurves.Name=xycurvename[counter]
        XYCurves.Npts=xycurve['npts']
        XYCurves.Xarray=xycurve['xarray'].tolist()
        XYCurves.Yarray=xycurve['yarray'].tolist()
    return DSSObj, True