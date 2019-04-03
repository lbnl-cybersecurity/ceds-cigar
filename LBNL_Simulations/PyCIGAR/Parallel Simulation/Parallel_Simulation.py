# # A sample code to test the python OpenDSS interface for voltage regulators and capacitors which also includes the use of OpenDSSdirect.py library
# import opendssdirect as dss
#
#
# dss.Basic.ClearAll()
#
# dss.Text.Command('Set parallel=no')
# # starts with actor 1
# dss.Text.Command('compile (C:/ceds-cigar/LBNL_Simulations/PyCIGAR/Parallel Simulation/Radial34Bus.DSS)')
# dss.Text.Command('set CPU=0')
# dss.Text.Command('solve')
# dss.Text.Command('NewActor')
# dss.Text.Command('compile (C:/ceds-cigar/LBNL_Simulations/PyCIGAR/Parallel Simulation/33BusMeshed.DSS)')
# dss.Text.Command('set CPU=1')
# dss.Text.Command('solve')
# dss.Text.Command('set parallel=yes')
# dss.Text.Command('SolveAll')
# dss.Text.Command('Wait')
# dss.Text.Command('set ConcatenateReports=Yes')
# dss.Text.Command('set activeactor=1')  # activates actor 1
# dss.Text.Command('show voltages')
# dss.Text.Command('set activeactor=2')  # activates actor 1
# dss.Text.Command('show voltages')


def DSSStartup():
    import win32com.client
    try:
        DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
        DSSText = DSSObj.Text
        DSSCircuit = DSSObj.ActiveCircuit
        DSSSolution = DSSCircuit.Solution
        DSSElem = DSSCircuit.ActiveCktElement
        DSSBus = DSSCircuit.ActiveBus
        DSSText.command = 'clear'
    except Exception as inst:
        print(type(inst))
        return {}
    return {'dssobj': DSSObj, 'dsstext': DSSText, 'dsscircuit': DSSCircuit, 'dsssolution': DSSSolution, 'dsselem': DSSElem, 'dssbus': DSSBus}


result = DSSStartup()
DSSText = result['dsstext']

DSSText.command = 'clear all'
DSSText.command = 'set parallel=no'
# starts with actor 1
DSSText.command = 'compile (C:/ceds-cigar/LBNL_Simulations/PyCIGAR/Parallel Simulation/Radial34Bus.DSS)'
DSSText.command = 'set CPU=0'
DSSText.command = 'solve'
DSSText.command = 'NewActor'
DSSText.command = 'compile (C:/ceds-cigar/LBNL_Simulations/PyCIGAR/Parallel Simulation/33BusMeshed.DSS)'
DSSText.command = 'set CPU=1'
DSSText.command = 'solve'
DSSText.command = 'set parallel=yes'
DSSText.command = 'SolveAll'
DSSText.command = 'Wait'
DSSText.command = 'set ConcatenateReports=Yes'
DSSText.command = 'set activeactor=1'  # activates actor 1
DSSText.command = 'show voltages'
DSSText.command = 'set activeactor=2'  # activates actor 1
DSSText.command = 'show voltages'
