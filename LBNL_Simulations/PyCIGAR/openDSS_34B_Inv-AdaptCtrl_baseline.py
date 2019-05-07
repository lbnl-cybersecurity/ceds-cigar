import numpy as np
import opendssdirect as dss
from utils.device.Inverter import Inverter
from utils.controller.AdaptiveInvController import AdaptiveInvController
from utils.controller.FixedInvController import FixedInvController
import matplotlib.pyplot as plt
from math import tan, acos
import copy
import pandas as pd
import random
from scipy.interpolate import interp1d

# Global variable initialization and error checking

mva_base = 1
load_scaling_factor = 1.5
generation_scaling_factor = 2.5
slack_bus_voltage = 1.04
noise_multiplier = 0
start_time = 42900  # Set simulation analysis period - the simulation is from StartTime to EndTime
end_time = 44000
end_time += 1  # creating a list, last element does not count, so we increase EndTime by 1
# Set hack parameters
time_step_of_hack = 500
percent_hacked = np.array([0, 0, 0, 0, 0, 0, 0, .5, 0, 0, .5, .5, .5, .5, .5, 0, 0, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Set initial VBP parameters for un-compromised inverters
VQ_start = 0.98
VQ_end = 1.01
VP_start = 1.02
VP_end = 1.05
hacked_settings = np.array([1.0, 1.001, 1.001, 1.01])
# Set delays for each node
Delay_VBPCurveShift = np.array([0, 0, 0, 0, 0, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Set observer voltage threshold
threshold_vqvp = 0.25
power_factor = 0.9
pf_converted = tan(acos(power_factor))
number_of_inverters = 13  # even feeder is 34Bus, we only have 13 inverters, which is chosen randomly for now

# File directory
FileDirectoryBase = '../Data Files/testpvnum10/'  # Get the data from the Testpvnum folder
network_model_directory = 'feeder/feeder34_B_NR/feeder34_B_NR.dss'

# Error checking of the global variable
if end_time < start_time or end_time < 0 or start_time < 0:
    print('Setup Simulation Times Inappropriately.')
if noise_multiplier < 0:
    noise_multiplier = 0
    print('Setup Noise Multiplier Correctly.')

# Global variable initialization done

dss.run_command('Redirect ' + network_model_directory)  # redirecting to the model
dss.Vsources.PU(slack_bus_voltage)  # setting up the slack bus voltage
# Setting up the solution parameters, check OpenDSS documentation for details
dss.Monitors.ResetAll()
dss.Solution.Mode(1)
dss.Solution.Number(1)
dss.Solution.StepSize(1)
dss.Solution.ControlMode(-1)
dss.Solution.MaxControlIterations(1000000)
dss.Solution.MaxIterations(30000)
dss.Solution.Solve()  # solve commands execute the power flow
if not dss.Solution.Converged:
    print('Initial Solution Not Converged. Check Model for Convergence')
    raise SystemError
else:
    print('Initial Model Converged. Proceeding to Next Step.')
    total_loads = dss.Loads.Count()
    all_load_names = dss.Loads.AllNames()
    print('OpenDSS Model Compilation Done.')

#####################################################################################################################################
# Load data from file
TimeResolutionOfData = 10  # resolution in minute
QSTS_Time = list(range(1441))  # This can be changed based on the available data - for example, 1440 time-steps
QSTS_Data = np.zeros((len(QSTS_Time), 4, total_loads))  # 4 columns as there are four columns of data available in the .mat file

total_number_of_files = 37  # number of load profiles currently available
# The idea is to make sure all load profiles are used once minimum
if total_loads <= total_number_of_files:
    file_index = list(range(1, total_loads + 1))
else:
    file_index = list(range(1, total_number_of_files + 1))  # Because the first file starts with 1, range always excludes the last number
    remaining_loads = [random.randint(1, total_number_of_files) for r in range(total_loads - total_number_of_files)]
    file_index += remaining_loads

    # file_index = [random.randint(1, total_number_of_files) for r in range(TotalLoads-total_number_of_files)] # Uncomment this line if total randomization is required for all the loads

for node in range(total_loads):
    # This is created manually according to the naming of the folder
    FileDirectoryExtension = 'node_' + str(file_index[node]) + '_pv_' + str(TimeResolutionOfData) + '_minute.csv'
    # The total file directory
    FileName = FileDirectoryBase + FileDirectoryExtension
    # Load the file
    MatFile = np.genfromtxt(FileName, delimiter=',')
    QSTS_Data[:, :, node] = MatFile  # Putting the loads to appropriate nodes according to the loadlist

Generation = QSTS_Data[:, 1, :] * generation_scaling_factor  # solar generation
Load = QSTS_Data[:, 3, :] * load_scaling_factor  # load demand
Generation = np.squeeze(Generation) / mva_base  # To convert to per unit, it should not be multiplied by 100
Load = np.squeeze(Load) / mva_base
print('Reading Data for Pecan Street is done.')

#####################################################################################################################################
# Interpolate to change data from minutes to seconds
print('Starting Interpolation...')
# interpolation for the whole period...
Time = list(range(start_time, end_time))
TotalTimeSteps = len(Time)
LoadSeconds = np.empty([3600 * 24, total_loads])
GenerationSeconds = np.empty([3600 * 24, total_loads])
# Interpolate to get minutes to seconds
for node in range(total_loads):  # i is node
    t_seconds = np.linspace(1, len(Load[:, node]), int(3600 * 24 / 1))
    f = interp1d(range(len(Load[:, node])), Load[:, node], kind='cubic', fill_value="extrapolate")
    LoadSeconds[:, node] = f(t_seconds)  # spline method in matlab equal to Cubic Spline -> cubic

    f = interp1d(range(len(Generation[:, node])), Generation[:, node], kind='cubic', fill_value="extrapolate")
    GenerationSeconds[:, node] = f(t_seconds)

# We take out only the window we want...
LoadSeconds = LoadSeconds[start_time:end_time, :]
GenerationSeconds = GenerationSeconds[start_time:end_time, :]
Load = LoadSeconds
Generation = GenerationSeconds
timeList = list(range(TotalTimeSteps))
print('Finished Interpolation!')

# Create noise vector
Noise = np.empty([TotalTimeSteps, total_loads])
for node in range(total_loads):
    Noise[:, node] = np.random.randn(TotalTimeSteps)

# Add noise to loads
for node in range(total_loads):
    Load[:, node] = Load[:, node] + noise_multiplier * Noise[:, node]

if noise_multiplier > 0:
    print('Load Interpolation has been done. Noise was added to the load profile.')
else:
    print('Load Interpolation has been done. No Noise was added to the load profile.')

#####################################################################################################################################

MaxGenerationPossible = np.max(Generation, axis=0)
sbar = MaxGenerationPossible

""" 
 nodes' variable is a dictionary contains all the nodes in the grid,
 with KEY is the node's number, VALUE is a dataframe with 
 ROW is ['Voltage', 'Generation', 'P', 'Q'] - Voltage is Voltage for each timestep; 
                                              P,Q is P,Q injection at that node for that timestep.
 and 
 COLUMN is each timestep for the whole simulation. 
 nodes = {
    1: dataFrame1,
    2: dataFrame2
 }
"""
nodes = {}
features = ['Voltage', 'Generation', 'P', 'Q']

for i in range(len(all_load_names)):
    df = pd.DataFrame(columns=list(range(TotalTimeSteps)), index=features)
    nodes[i] = df
    nodes[i].loc['Generation'] = Generation[:, i]
    nodes[i].loc['P'] = 0
    nodes[i].loc['Q'] = 0

# INITIALIZE INVERTERS
#####################################################################################################################################
""" 
 inverters' variable is a dictionary contains all the inverters in the grid,
 with KEY is the node's number where we have inverters, VALUE is a list of inverters at that node.
 
 Each inverter has a dictionary:
     'device': Inverter_Object
     'controller: Controller_Object
     'info': contain scenario information in a dataframe (solar generation, sbar) for each timestep at that node
 }
"""

inverters = {}
features = ['Generation', 'sbar']

# we create inverters from node 5 to node (5+13), the offset is just a chosen value
offset = 5

for i in range(len(all_load_names)):
    inverters[i] = []
    if offset - 1 < i < number_of_inverters + offset:
        # inverter device
        inv = {}
        inv['device'] = Inverter(timeList, lpf_meas=1, lpf_output=0.1)
        # controller: timeList, VBP is initial VBP, delayTimer is the delay control on VBP
        inv['controller'] = AdaptiveInvController(timeList, VBP=np.array([VQ_start, VQ_end, VP_start, VP_end]), delayTimer=Delay_VBPCurveShift[i], device=inv['device'])
        # prepare info
        df = pd.DataFrame(columns=list(range(TotalTimeSteps)), index=features)
        df.loc['Generation'] = Generation[:, i]
        df.loc['sbar'] = sbar[i]
        timeList = list(range(TotalTimeSteps))
        inv['info'] = df
        inverters[i].append(inv)

#####################################################################################################################################

# for each time-step in the simulation
for timeStep in range(TotalTimeSteps):

    # run the power flow
    # for the first steps, we just initialize voltage value, no pq injection
    if timeStep == 0:
        for node in range(len(all_load_names)):
            nodeName = all_load_names[node]
            dss.Loads.Name(nodeName) # set active load
            dss.Loads.kW(Load[timeStep, node])
            dss.Loads.kvar(pf_converted * Load[timeStep, node])
    # otherwise, we add Active Power (P) and Reactive Power (Q) which we injected at last time-step to the grid at that node
    else:
        for node in range(len(all_load_names)):
            nodeName = all_load_names[node]
            dss.Loads.Name(nodeName)  # set active load
            dss.Loads.kW(Load[timeStep, node] + nodes[node].at['P', timeStep - 1])
            dss.Loads.kvar(pf_converted * Load[timeStep, node] + nodes[node].at['Q', timeStep - 1])

    # solve() openDSS with new values of Load
    dss.Solution.Solve()
    if not dss.Solution.Converged:
        print('Solution Not Converged at Step:', timeStep)

    nodeInfo = []
    for nodeName in all_load_names:
        dss.Circuit.SetActiveElement('load.' + nodeName) # set active element
        dss.Circuit.SetActiveBus(dss.CktElement.BusNames()[0]) # grab the bus for the active element
        voltage = dss.Bus.puVmagAngle()[::2] # get the pu information directly
        nodeInfo.append(np.mean(voltage)) # average of the per-unit voltage

    # distribute voltage to node
    for i in range(len(nodes)):
        node = nodes[i]
        node.at['Voltage', timeStep] = nodeInfo[i]

    #####################################################################################################################################

    # at hack time-step, we do this
    if timeStep == time_step_of_hack - 1:
        # with each node...
        for node in range(len(all_load_names)):
            # if we have inverters at that node...
            if inverters[node] != []:
                # we get the first inverter in the list...
                inverter = inverters[node][0]

                # we create a new inverter, called hacked inverter, this controller is FixedInvController
                # that the Controller doesnt change the VBP w.r.t timestep
                hackedInv = copy.deepcopy(inverter)
                for k in range(timeStep, TotalTimeSteps):
                    hackedInv['controller'] = FixedInvController(timeList, VBP=hacked_settings, device=hackedInv)

                # change the sbar, generation of hacked Inverter Info, control percentHacked
                hackedInv['info'].loc['sbar'][timeStep:] = hackedInv['info'].loc['sbar'][timeStep:] * percent_hacked[node]
                hackedInv['info'].loc['Generation'][timeStep:] = hackedInv['info'].loc['Generation'][timeStep:] * percent_hacked[node]
                # add the hacked inverter into the list of inverters at that node
                inverters[node].append(hackedInv)
                # generation and sbar change on the original inverter, only control 1-percentHacked
                inverter['info'].loc['sbar'][timeStep:] = inverter['info'].loc['sbar'][timeStep:] * (1 - percent_hacked[node])
                inverter['info'].loc['Generation'][timeStep:] = inverter['info'].loc['Generation'][timeStep:] * (1 - percent_hacked[node])
    ################################################################################################################################################

    # with each node in the grid...
    for node in range(len(all_load_names)):
        # if we have inverters at that node then...
        if inverters[node] != []:
            invertersNode = inverters[node]  # get the list of inverters at that node
            for inverter in invertersNode:  # get an inverter at that node
                device = inverter['device']
                controller = inverter['controller']
                info = inverter['info']
                # Calculate P Q injected into the grid by an inverter
                p_inv, q_inv = device.step(v=nodes[node].at['Voltage', timeStep], solar_irr=info.at['Generation', timeStep], solar_minval=5, Sbar=info.at['sbar', timeStep], VBP=controller.get_VBP())

                # add P Q injected to the node
                nodes[node].at['P', timeStep] += p_inv
                nodes[node].at['Q', timeStep] += q_inv

                # change internal VBP
                controller.act(nk=0.1, device=device, thresh=threshold_vqvp)
    ################################################################################################################################################

#  Plotting
f = plt.figure(figsize=[20, 10])
for node in nodes:
    plt.plot(nodes[node].loc['Voltage'])
plt.title('Voltage at Inverter Nodes')
plt.show()

f = plt.figure(figsize=[20, 10])
for i in range(5, 18):
    x = inverters[i][0]['controller'].VBP
    y = np.zeros([len(x), x[0].shape[0]])
    for i in range(len(x)):
        y[i, :] = x[i]
    plt.plot(y[:, 0], 'r')
    plt.plot(y[:, 1], 'y')
    plt.plot(y[:, 2], 'b')
    plt.plot(y[:, 3], 'k')
plt.title('Movement of VBP points')
plt.show()

f = plt.figure(figsize=[20, 10])
plt.plot(nodes[2].loc['Generation'], marker='o')
plt.title('Generation at a chosen Node')
plt.show()
################################################################################################################################################
