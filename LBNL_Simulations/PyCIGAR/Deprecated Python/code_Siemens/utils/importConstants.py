# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:30:36 2019

@author: shamm
"""
from math import tan, acos
import numpy as np

Sbase = 1
load_scaling_factor = 1.5
generation_scaling_factor = 5
slack_bus_voltage = 1.04
noise_multiplier = 0
# Set simulation analysis period - the simulation is from StartTime to EndTime
start_time = 42900
end_time = 44000
end_time += 1  # creating a list, last element does not count, so we increase EndTime by 1
# Set hack parameters
TimeStepOfHack = 300
percent_hacked = np.array([0, 0, 0, 0, 0, 0, 0, .5, 0, 0, .5, .5, .5, .5, .5, 0, 0, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Set delays for each node
Delay_VBPCurveShift = np.array([0, 0, 0, 0, 0, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# number_of_Inverters = 13

# Set observer voltage threshold
ThreshHold_vqvp = 0.25
power_factor = 0.9
pf_converted = tan(acos(power_factor))
Number_of_Inverters = 13  # even feeder is 34Bus, we only have 13 inverters

# Error checking of the global variable -- TODO: add error handling here!
if end_time < start_time or end_time < 0 or start_time < 0:
    print('Setup Simulation Times Inappropriately.')
if noise_multiplier < 0:
    print('Setup Noise Multiplier Correctly.')

IEEE123_model_directory = 'C:\\ceds-cigar\\LBNL_Simulations\\PyCIGAR\\feeder\\IEEE123Bus\\IEEE123Master.dss'
