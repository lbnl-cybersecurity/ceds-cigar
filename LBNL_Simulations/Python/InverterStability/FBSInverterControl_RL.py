import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import tan,acos
import os
import matlab.engine as matlab

def start_matlab():
    return matlab.start_matlab()
def quit_matlab(matlab_engine):
    matlab_engine.quit()
def FBS(matlab_engine, NodeVoltageToPlot, SlackBusVoltage, IncludeSolar):
    return matlab_engine.FBS(NodeVoltageToPlot,SlackBusVoltage,IncludeSolar,nargout=2)
def ieee_feeder_mapper(matlab_engine, IeeeFeeder):
    return matlab_engine.ieee_feeder_mapper(IeeeFeeder)

LoadScalingFactor=0.9*1.5
GenerationScalingFactor=2
SlackBusVoltage=1.02
power_factor=0.9
IncludeSolar=1

IeeeFeeder = 13
matlab_engine=start_matlab()

print(len(ieee_feeder_mapper(matlab_engine, IeeeFeeder)))


