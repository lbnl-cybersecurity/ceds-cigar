import opendssdirect as DSSObj
import numpy as np
from scipy.interpolate import interp1d
from math import tan,acos
import copy
import pandas as pd
import time
#######################################################
####         Load data from file                    ###
#######################################################
#######################################################
def ReadScenarioFile(FileDirectory='/home/toanngo/Documents/GitHub/ceds-cigar/LBNL_Simulations/testpvnum10/',
                    OpenDSSDirectory='/home/toanngo/Documents/GitHub/ceds-cigar/LBNL_Simulations/feeders/feeder34_B_NR/feeder34_B_NR.dss'):
    DSSObj.run_command('Redirect ' + OpenDSSDirectory)
    TotalLoads=DSSObj.Loads.Count()
    AllLoadNames=DSSObj.Loads.AllNames()
    Sbase=1
    LoadScalingFactor = 1.5
    GenerationScalingFactor = 5
    #Retrieving the data from the load profile
    TimeResolutionOfData=10 #resolution in minute
    #Get the data from the Testpvnum folder
    #Provide Your Directory - move testpvnum10 from github to drive C: 
    QSTS_Time = list(range(1441)) #This can be changed based on the available data - for example, 1440 timesteps
    QSTS_Data = np.zeros((len(QSTS_Time),4,TotalLoads)) #4 columns as there are four columns of data available in the .mat file

    for node in range(TotalLoads):
        #This is created manually according to the naming of the folder
        FileDirectoryExtension = 'node_' + str(node+1) + '_pv_' +str(TimeResolutionOfData) + '_minute.csv'
        #The total file directory
        FileName = FileDirectory + FileDirectoryExtension
        #Load the file
        MatFile = np.genfromtxt(FileName, delimiter=',')    
        QSTS_Data[:,:,node] = MatFile #Putting the loads to appropriate nodes according to the loadlist

    Generation = QSTS_Data[:,1,:]*GenerationScalingFactor #solar generation
    Load = QSTS_Data[:,3,:]*LoadScalingFactor #load demand
    RawGeneration = np.squeeze(Generation)/Sbase  #To convert to per unit, it should not be multiplied by 100
    RawLoad = np.squeeze(Load)/Sbase
    print('Reading Data for Pecan Street is done.')
    
    return RawLoad, RawGeneration, TotalLoads, AllLoadNames 


def ScenarioInterpolation(startTime, endTime, RawLoad, RawGeneration):
    print('Starting Interpolation...')
    #interpolation for the whole period...
    Time = list(range(startTime,endTime))
    TotalTimeSteps = len(Time)
    TotalLoads = RawLoad.shape[1]
    
    LoadSeconds = np.empty([3600*24, TotalLoads])
    GenerationSeconds = np.empty([3600*24, TotalLoads])
    # Interpolate to get minutes to seconds
    for node in range(TotalLoads): # i is node
        t_seconds = np.linspace(1,len(RawLoad[:,node]), int(3600*24/1))
        f = interp1d(range(len(RawLoad[:,node])), RawLoad[:,node], kind='cubic', fill_value="extrapolate")
        LoadSeconds[:,node] = f(t_seconds) #spline method in matlab equal to Cubic Spline -> cubic

        f = interp1d(range(len(RawGeneration[:,node])), RawGeneration[:,node], kind='cubic', fill_value="extrapolate")
        GenerationSeconds[:,node]= f(t_seconds)

    # Initialization
    # then we take out only the window we want...
    LoadSeconds = LoadSeconds[startTime:endTime,:]
    GenerationSeconds = GenerationSeconds[startTime:endTime,:]
    return LoadSeconds, GenerationSeconds

def ScenarioAddNoise(Load, Generation, TotalLoads, addNoise=True):
    TotalTimeSteps = len(Load)
    if addNoise == True:
        NoiseMultiplyer = 1
    else:
        NoiseMultiplyer = 0
    Noise = np.empty([TotalTimeSteps, TotalLoads])
    for node in range(TotalLoads):
        Noise[:,node] = np.random.randn(TotalTimeSteps) 

    #Add noise to loads
    for node in range(TotalLoads):
        Load[:,node] = Load[:,node] + NoiseMultiplyer*Noise[:,node]

    if NoiseMultiplyer > 0:
        print('Load Interpolation has been done. Noise was added to the load profile.') 
    else:
        print('Load Interpolation has been done. No Noise was added to the load profile.') 
    return Load, Generation