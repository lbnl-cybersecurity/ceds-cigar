import numpy as np
from DSSStartup import DSSStartup
from setInfo import *
from getInfo import *
import matplotlib.pyplot as plt
from math import tan,acos

	################################
	########GLOBAL INIT#############
	################################
def Main():
	Sbase=1
	LoadScalingFactor = 7
	GenerationScalingFactor = 5 
	SlackBusVoltage = 1.05 
	NoiseMultiplyer= 0

	#Set simulation analysis period - the simulation is from StartTime to EndTime
	StartTime = 40000 
	EndTime = 40500
	EndTime += 1 # creating a list, last element does not count, so we increase EndTime by 1
	#Set hack parameters
	TimeStepOfHack = 50

	PercentHacked = np.array([0.5, 0, 1.0, 0, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0.5, 0,
							0, 0, 1.0, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0.5, 0, 0.5, 1, 0])

	#Set adaptive controller gain values (the higher the gain, the faster the response time)
	kq = 1
	kp = 1

	#Set delays for each node
	                        #1  2  3   4   5  6   7   8      9    10   11   12   13
	Delay_VoltageSampling = np.array([0, 0, 0, 0, 0, 0, 0, 10, 0, 0,  10,  10,  50,  10,  10,  10,  10,  10, 0,
	                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
	Delay_VBPCurveShift =   np.array([0, 0, 0, 0, 0, 0, 0, 120, 0, 0, 120, 120, 120, 120, 120, 120, 120, 120, 0,
	                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	#Set observer voltage threshold
	ThreshHold_vqvp = 0.25
	power_factor=0.9
	pf_converted=tan(acos(power_factor))
	Number_of_Inverters = 13 #even feeder is 34Bus, we only have 13 inverters

	#The following variable allows to run the simulation without any inverter
	SimulateInverterHack=1

	#Error checking of the global variable -- TODO: add error handling here!
	if EndTime < StartTime or EndTime < 0 or StartTime < 0:
	    print('Setup Simulation Times Appropriately.')
	if NoiseMultiplyer < 0:
	    print('Setup Noise Multiplyer Correctly.')

	################################
	########OPENDSS INIT############
	################################
	DSSStart = DSSStartup()
	DSSText =DSSStart['dsstext']
	DSSSolution = DSSStart['dsssolution']
	DSSCircuit = DSSStart['dsscircuit']
	DSSObj = DSSStart['dssobj']
	DSSMon = DSSCircuit.Monitors
	DSSText.command = 'Compile C:\\feeders\\feeder34_B_NR\\feeder34_B_NR.dss'
	    
	DSSSolution.Solve();
	if not DSSSolution.Converged:
	    print('Initial Solution Not Converged. Check Model for Convergence')
	else:
	    print('Initial Model Converged. Proceeding to Next Step.')
	    #Doing this solve command is required for GridPV, that is why the monitors
	    #go under a reset process
	    DSSMon.ResetAll
	    setSolutionParams(DSSObj,'daily',1,1,'off',1000000,30000)
	    #Easy process to get all names and count of loads, a trick to avoid
	    #some more lines of code
	    TotalLoads=DSSCircuit.Loads.Count
	    AllLoadNames=DSSCircuit.Loads.AllNames
	    print('OpenDSS Model Compliation Done.')

	################################
	########LOAD DATA   ############
	################################
	#Retrieving the data from the load profile
	TimeResolutionOfData=10 #resolution in minute
	#Get the data from the Testpvnum folder
	#Provide Your Directory - move testpvnum10 from github to drive C: 
	FileDirectoryBase ='C:\\feeders\\testpvnum10\\';
	QSTS_Time = list(range(1441)) #This can be changed based on the available data - for example, 1440 timesteps
	QSTS_Data = np.zeros((len(QSTS_Time) ,4,TotalLoads)) #4 columns as there are four columns of data available in the .mat file

	for node in range(TotalLoads):
	    #This is created manually according to the naming of the folder
	    FileDirectoryExtension = 'node_' + str(node+1) + '_pv_' +str(TimeResolutionOfData) + '_minute.csv'
	    #The total file directory
	    FileName = FileDirectoryBase + FileDirectoryExtension
	    #Load the file
	    MatFile = np.genfromtxt(FileName, delimiter=',')    
	    QSTS_Data[:,:,node] = MatFile #Putting the loads to appropriate nodes according to the loadlist
	    
	Generation = QSTS_Data[:,1,:]*GenerationScalingFactor #solar generation
	Load = QSTS_Data[:,3,:]*LoadScalingFactor #load demand
	Generation = np.squeeze(Generation)/Sbase  #To convert to per unit, it should not be multiplied by 100
	Load = np.squeeze(Load)/Sbase
	print('Reading Data for Pecan Street is done.')

	###########################################################
	########INTERPOLATE DATA FROM MINUTE TO SECOND#############
	###########################################################
	from scipy.interpolate import interp1d

	print('Starting Interpolation...')

	#interpolation for the whole period...
	Time = list(range(StartTime,EndTime))
	TotalTimeSteps = len(Time)
	LoadSeconds = GenerationSeconds = np.empty([3600*24, TotalLoads])
	# Interpolate to get minutes to seconds
	for node in range(TotalLoads): # i is node
	    t_seconds = np.linspace(1,len(Load[:,node]), int(3600*24/1))
	    f = interp1d(range(len(Load[:,node])), Load[:,node], kind='cubic', fill_value="extrapolate")
	    LoadSeconds[:,node] = f(t_seconds) #spline method in matlab equal to Cubic Spline -> cubic
	    
	    f = interp1d(range(len(Generation[:,node])), Generation[:,node], kind='cubic', fill_value="extrapolate")
	    GenerationSeconds[:,node]= f(t_seconds)

	# Initialization
	# then we take out only the window we want...
	LoadSeconds = LoadSeconds[StartTime:EndTime,:]
	GenerationSeconds = GenerationSeconds[StartTime:EndTime,:]
	Load = LoadSeconds
	Generation = GenerationSeconds

	#Create noise vector
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
	    
	MaxGenerationPossible = np.max(Generation, axis = 0)
	SolarGeneration_vqvp = Generation * GenerationScalingFactor


	Sbar_max =  MaxGenerationPossible * GenerationScalingFactor
	Sbar_m = {}
	hacked = {}
	delay_sampling = {}
	delay_shift = {}
	for i in range(len(AllLoadNames)):
	    name = AllLoadNames[i]
	    hacked[name] = PercentHacked[i]
	    delay_sampling[name] = Delay_VoltageSampling[i]
	    delay_shift[name] = Delay_VBPCurveShift[i]
	    Sbar_m[name] = Sbar_max[i]

	PercentHacked = hacked
	Delay_VoltageSampling = delay_sampling
	Delay_VBPCurveShift = delay_shift
	Sbar_max = Sbar_m

	VQ_start = 1.01 
	VQ_end = 1.03 
	VP_start = 1.03 
	VP_end = 1.05
	VBPUnhacked = np.array([VQ_start, VQ_end, VP_start, VP_end])
	VQ_startHacked = 1.01 
	VQ_endHacked = 1.015 
	VP_startHacked = 1.015 
	VP_endHacked = 1.02
	VBPHacked = np.array([VQ_startHacked, VQ_endHacked, VP_startHacked, VP_endHacked])

	return DSSObj, DSSSolution, Load, Generation, Sbar_max
