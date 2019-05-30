'''
This code is an example of how to open and close lines in OpenDSS using the DSSText Command.
'''

import opendssdirect as dss
import numpy as np

# The DSS object holds the entire OpenDSS model
# dss.run_command('Redirect feeder/33BusMeshed/33BusMeshed.dss')
dss.run_command('Redirect 33BusMeshed.dss')
# Solving the power flow which is a snapshot power flow by default
dss.Solution.Solve()

'''
    Printing Names of all the loads 
'''
# Selecting the first load
dss.Loads.First()

while True:
    print(f'Name of the Load: {dss.Loads.Name()}')
    if not dss.Loads.Next() > 0:
        break

'''
 Get the voltage of a selected load, change its rating, rerun the power flow, and retrieve the information again
'''

# Assume we want to change the KW and KVAR value by 10% for a specific load (Load 32)

# Activating the load
selected_load = 'load 33'
dss.Loads.Name(selected_load)

# Changing the Vmin per unit, if the load voltage goes below this, the model will be converted to a constant Z model
dss.Loads.Vminpu(0.8)
# Assigning the load model to be a constant power model
dss.Loads.Model(1)
# Set the load as the active element
dss.Circuit.SetActiveElement('load.' + selected_load)  # set active element
dss.Circuit.SetActiveBus(dss.CktElement.BusNames()[0])  # grab the bus for the active element
voltage = dss.Bus.puVmagAngle()[::2]  # get the pu information directly
print(f'Average Per unit voltage of the load : {np.mean(voltage)}')
# retrieving the load kw and kvar
kw, kvar = dss.Loads.kW(), dss.Loads.kvar()
# Set the kw and kvar value for the selected load, increasing the kw and kvar values by 50%
dss.Loads.kW(1.5 * kw)
dss.Loads.kvar(1.5 * kvar)
# Solving the power flow again
dss.Solution.Solve()
dss.Circuit.SetActiveElement('load.' + selected_load)  # set active element
dss.Circuit.SetActiveBus(dss.CktElement.BusNames()[0])  # grab the bus for the active element
voltage = dss.Bus.puVmagAngle()[::2]  # get the pu information directly
print('Average Per unit voltage of the load after 10% increase: {}'.format(np.mean(voltage)))

voltage_list = []

'''
    Changing the load rating iteratively, this is kind of a QSTS simulation, if the index represents time
'''
for i in range(0, 11):
    selected_load = 'load 33'
    dss.Loads.Name(selected_load)
    # Change the load kw and kvar value, by ramping the original value
    dss.Loads.kW((1 + i / 10) * kw)
    dss.Loads.kvar((1 + i / 10) * kvar)
    # Solving the power flow
    dss.Solution.Solve()
    dss.Circuit.SetActiveElement('load.' + selected_load)  # set active element
    dss.Circuit.SetActiveBus(dss.CktElement.BusNames()[0])  # grab the bus for the active element
    voltage = dss.Bus.puVmagAngle()[::2]  # get the pu information directly

    # retrieving the load kw and kvar
    voltage_list.append(np.mean(voltage))
    print('Average Per unit voltage of the load at {} iteration is : {}'.format(i, voltage_list[i]))

