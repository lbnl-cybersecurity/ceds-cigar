def setLoadInfo(DSSObj,loadname,property,value,NameChecker):
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

def setSourceInfo(DSSObj,sourcename,property,value,NameChecker):
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