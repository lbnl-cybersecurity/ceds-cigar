from replay_buffer import ReplayBuffer
from ornstein_uhlenbeck_noise import OrnsteinUhlenbeckActionNoise
from ddpg_agent_tf import Agent
from plotting import plotting
from env import env

import pandas as pd
import numpy as np
from math import tan,acos
import os
  

################################################################################
######################## Init global variables #################################
################################################################################
LoadScalingFactor=2000
GenerationScalingFactor=50
SlackBusVoltage=1.02
power_factor=0.9
IncludeSolar=1

LineNames = ('l_632_633','l_632_645','l_632_671','l_633_634','l_645_646','l_650_632','l_671_680','l_671_684',
             'l_671_692','l_684_611','l_684_652','l_692_675','l_u_650')

AllBusNames = ('sourcebus',
               'load_611','load_634','load_645','load_646','load_652','load_671','load_675','load_692',
               'bus_611','bus_634','bus_645','bus_646','bus_652','bus_671','bus_675','bus_692','bus_632',
               'bus_633','bus_650','bus_680','bus_684')

LoadBusNames = AllBusNames[1:9]
BusNames = AllBusNames[9:22]
IeeeFeeder = 13

LoadList = np.array([6,7,8,13,3,12,11,10])-1
NodeList = np.array([650,632,671,680,633,634,645,646,684,611,692,675,652])
BusesWithControl = NodeList[LoadList]

NumberOfLoads=len(LoadBusNames)
NumberOfNodes=len(BusNames)

FeederMap = np.array(
      [[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]])

Z_in_ohm = np.array(
      [[0.00000000e+00+0.00000000e+00j],
       [1.31250000e-01+3.85568182e-01j],
       [1.31250000e-01+3.85568182e-01j],
       [6.56250000e-02+1.92784091e-01j],
       [7.12689394e-02+1.11875000e-01j],
       [1.42537879e-05+2.23750000e-05j],
       [1.25890152e-01+1.27566288e-01j],
       [7.55340909e-02+7.65397727e-02j],
       [7.52159091e-02+7.70965909e-02j],
       [7.55227273e-02+7.65625000e-02j],
       [1.42537879e-05+2.23750000e-05j],
       [7.55871212e-02+4.22632576e-02j],
       [2.03409091e-01+7.76363636e-02j]])

Paths = np.array(
      [[ 1.,  2.,  3.,  4.,  0.],
       [ 1.,  2.,  5.,  6.,  0.],
       [ 1.,  2.,  7.,  8.,  0.],
       [ 1.,  2.,  3.,  9., 10.],
       [ 1.,  2.,  3., 11., 12.],
       [ 1.,  2.,  3.,  9., 13.]])

#Base value calculation
Vbase = 4.16e3 
Sbase = 1.0 
Zbase = Vbase**2/Sbase
Ibase = Sbase/Vbase
Z = Z_in_ohm/Zbase

#Load Data Pertaining to Loads to create a profile
PV_Feeder_model = 10
FileDirectoryBase = 'C:\\Users\\Sy-Toan\\ceds-cigar\\LBNL_Simulations\\testpvnum10\\'

Time = list(range(1441))
TotalTimeSteps = len(Time)

#preparing QSTS_Data
QSTS_Data = np.zeros((TotalTimeSteps,4,IeeeFeeder))
for node in range(NumberOfLoads):
    FileDirectoryExtenstion = 'node_' + str(node+1) + '_pv_' + str(PV_Feeder_model) + '_minute.csv'
    FileName = FileDirectoryBase + FileDirectoryExtenstion
    MatFile = np.genfromtxt(FileName, delimiter=',')
    QSTS_Data[:,:,int(LoadList[node])] = MatFile

#Seperate PV Generation Data
Generation = QSTS_Data[:,1,:]*GenerationScalingFactor
Load = QSTS_Data[:,3,:]*LoadScalingFactor
Generation = np.squeeze(Generation)/Sbase
Load = np.squeeze(Load)/Sbase
MaxGenerationPossible = np.max(Generation, axis=0)

#Voltage Observer Parameters and related variable initialization
LowPassFilterFrequency = 0.1
HighPassFilterFrequency = 1.0
Gain_Energy = 1e5
TimeStep = 1

#ZIP load modeling
ConstantImpedanceFraction = 0.2
ConstantCurrentFraction = 0.05
ConstantPowerFraction = 0.75
ZIP_demand = np.zeros((TotalTimeSteps,IeeeFeeder,3), dtype=np.complex)

for node in range(1,IeeeFeeder):
    ZIP_demand[:,node,:] = np.array([ConstantPowerFraction*Load[:,node].astype(complex), 
                            ConstantCurrentFraction*Load[:,node].astype(complex), 
                            ConstantImpedanceFraction*Load[:,node].astype(complex)]).T*(1 + 1j*tan(acos(power_factor)))
    
#Power Flow with QVQP Control Case
Sbar =  MaxGenerationPossible * GenerationScalingFactor

PowerEachTimeStep_vqvp = np.zeros((IeeeFeeder,3), dtype=np.complex)
SolarGeneration_vqvp = Generation * GenerationScalingFactor
InverterRateOfChangeLimit = 100 
InverterRateOfChangeActivate = 0

#Drop Control Parameters
VQ_start = 1.01
VQ_end = 1.015
VP_start = 1.015
VP_end = 1.02

VBP = np.full([IeeeFeeder, 4, TotalTimeSteps], np.nan)
VBP[:,0,0] = VQ_start
VBP[:,1,0] = VQ_end
VBP[:,2,0] = VP_start
VBP[:,3,0] = VP_end

InverterLPF = 1
ThreshHold_vqvp = 0.25
V0 = np.full((TotalTimeSteps, 1), SlackBusVoltage)
#Adaptive controller parameters

#Delays                 [  1   2   3*  4   5   6*  7*  8*  9* 10* 11* 12* 13*
Delay_VoltageSampling = [ 0,   0,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1]

######################################################################################
######################################################################################
######################################################################################
def train(sess, env, args, allAction, defAgent, attAgent):
    
    #set up defender and attacker
    defenseAgent = Agent(sess, agent_code=defAgent, attack=False)
    attackAgent = Agent(sess, agent_code=attAgent, attack=True)
    
    #for each episode,...
    for i in range(int(args['max_episodes'])):
    	#we reset the environment...
        s = env.reset()
        
        translate = np.array([0.9, 0.9, 0.9, 0.9]) #this is because we want action in range 0.9 to 1.05..
        										   #the output of actor network is from 0 -> 0.15, we translate every node value to 0.9
        #and for each timestep...
        for j in range(int(args['max_episode_len'])):
            if j == 0:
            	#if it is the begin of the episode, we reset agent action to default configuration...
                allAction[defAgent] = np.array([1.01,1.015,1.015,1.02])
                allAction[attAgent] = np.array([1.01,1.015,1.015,1.02])
                aDef = np.array([1.01,1.015,1.015,1.02])-translate
                aAtt = np.array([1.01,1.015,1.015,1.02])-translate
            else:
            	#otherwise we get the action from the actor network of that agent...
                aDef = defenseAgent.get_action(s)
                aAtt = attackAgent.get_action(s)
                #this allAction is the set of all VBP for all nodes to feed into the env...
                allAction[defAgent] = copy.deepcopy(aDef + translate)
                allAction[attAgent] = copy.deepcopy(aAtt + translate)
            
            #call the env to simulate what is going to happen if we are taking that action...
            s2, terminal, y_k, voltage = env.step(allAction)
            
            #then we add this experience into the buffer...
            defenseAgent.add_experience_to_buffer(s, aDef, y_k, terminal, s2)
            attackAgent.add_experience_to_buffer(s, aAtt, y_k, terminal, s2)
            
            #then we train our agents...
            defenseAgent.train()
            attackAgent.train()
            
            #set the current state 
            s = s2

            #this is to update the summaries of agent
            if not terminal:
                defenseAgent.summaries(terminal, voltage, allAction[defAgent])
                attackAgent.summaries(terminal, voltage, allAction[attAgent])
            else:
                sumVoltageDef, sumActionDef, sumRewardDef = defenseAgent.summaries(terminal, voltage, allAction[defAgent])
                sumVoltageAtt, sumActionAtt, sumRewardAtt = attackAgent.summaries(terminal, voltage, allAction[attAgent])
                print('Agent: {:d} | Defense | Reward: {:f} | Episode: {:d} '.format(defAgent, sumRewardDef[-1:][0],i))
                print('Agent: {:d} | Attack  | Reward: {:f} | Episode: {:d} '.format(attAgent, sumRewardAtt[-1:][0],i), end="")
                #plotting(sumVoltageDef, sumActionDef, sumRewardDef, sumVoltageAtt, sumActionAtt, sumRewardAtt)
                break
    return defenseAgent, attackAgent

#main body

def main():
    with tf.Session() as sess:

        simulation = env()        
        args={}
        args['state_dim'] = 9
        args['action_dim'] = 4
        args['action_bound'] = 0.15
        args['actor_lr'] = 0.0001
        args['critic_lr'] = 0.001
        args['gamma'] = 0.99
        args['tau'] = 0.001
        args['buffer_size'] = 5000
        args['minibatch_size'] = 50
        args['random_seed'] = 1234
        args['max_episodes'] = 150
        args['max_episode_len'] = TotalTimeSteps
        args['summary_dir'] = os.getcwd()
        
        #init 
        tf.set_random_seed(args['random_seed'])
        
        #set all VBP value from all nodes to the same values, DOS attack --
        VBP = np.full([IeeeFeeder, 4, TotalTimeSteps], np.nan)
        for i in range(1441):
            VBP[:,0,i] = VQ_start
            VBP[:,1,i] = VQ_end
            VBP[:,2,i] = VP_start
            VBP[:,3,i] = VP_end
        allAction = copy.deepcopy(VBP[:,:,0])
        
        defenseAgent = 2
        attackAgent = 5
        defAgent, attAgent = train(sess, simulation, args, allAction, defenseAgent, attackAgent)
        #test(sess, simulation, args, actor)
        return defAgent, attAgent