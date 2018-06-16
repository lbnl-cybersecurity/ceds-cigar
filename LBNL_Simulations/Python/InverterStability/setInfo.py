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
            print('Load Not Found')
            return DSSObj,False
    if (len(sourcename) != len(value)):
        return DSSObj,False
    for counter in range(len(value)):
        Sources.name=sourcename[counter]
        if (property=='pu'):
            Sources.pu = value[counter]
        elif (property=='basekv'):
            Sources.BasekV = value[counter]
        # elif (property=='kva'):
        #     Sources.kva = value[counter]
        # elif (property=='pf'):
        #     Sources.PF=value[counter]
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