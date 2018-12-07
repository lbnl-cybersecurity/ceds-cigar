import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import tan,acos
import os
import matlab
import matlab.engine

import tensorflow as tf
import tflearn

print("what could go wrong?")
def start_matlab():
    return matlab.engine.start_matlab()
def quit_matlab(matlab_engine):
    matlab_engine.quit()
    
def ieee_feeder_mapper(matlab_engine, IeeeFeeder):
    FeederMap, Z_in_ohm, Paths, NodeList, LoadList = matlab_engine.ieee_feeder_mapper(IeeeFeeder, nargout=5)
    FeederMap = np.array(FeederMap)
    Z_in_ohm = np.array(Z_in_ohm)
    Paths = np.array(Paths)
    NodeList = np.array(NodeList)
    LoadList = np.array(LoadList)
    return FeederMap, Z_in_ohm, Paths, NodeList[0], LoadList[0]-1

def FBSfun(matlab_engine, V0, loads, Z, B):
    V, _, S, _ = matlab_engine.FBSfun(float(V0), matlab.double(loads.tolist(),is_complex=True), matlab.double(Z.tolist(), is_complex=True), matlab.double(B.tolist(),is_complex=True), nargout=4)
    V = np.array(V, dtype=np.complex).squeeze()
    S = np.array(S, dtype=np.complex).squeeze()
    return V, S

IeeeFeeder = 13
matlab_engine = start_matlab()  

LoadScalingFactor=2000
GenerationScalingFactor=50
SlackBusVoltage=1.02
power_factor=0.9
IncludeSolar=1

#Feeder parameters

LineNames = ('l_632_633','l_632_645','l_632_671','l_633_634','l_645_646','l_650_632','l_671_680','l_671_684',
             'l_671_692','l_684_611','l_684_652','l_692_675','l_u_650')

AllBusNames = ('sourcebus',
               'load_611','load_634','load_645','load_646','load_652','load_671','load_675','load_692',
               'bus_611','bus_634','bus_645','bus_646','bus_652','bus_671','bus_675','bus_692','bus_632',
               'bus_633','bus_650','bus_680','bus_684')

LoadBusNames = AllBusNames[1:9]
BusNames = AllBusNames[9:22]
IeeeFeeder = 13
print("yay")
LoadList = np.array([6,7,8,13,3,12,11,10])-1
NodeList = np.array([650,632,671,680,633,634,645,646,684,611,692,675,652])
BusesWithControl = NodeList[LoadList]
print("yay")
NumberOfLoads=len(LoadBusNames)
NumberOfNodes=len(BusNames)

FeederMap, Z_in_ohm, Paths, _, _ = ieee_feeder_mapper(matlab_engine, IeeeFeeder)

#Base value calculation
Vbase = 4.16e3 #4.16 kV
Sbase = 1.0 #500 kVA
Zbase = Vbase**2/Sbase
Ibase = Sbase/Vbase
Z = Z_in_ohm/Zbase

#Load Data Pertaining to Loads to create a profile
PV_Feeder_model = 10
FileDirectoryBase = 'C:\\Users\\Toan Ngo\\Documents\\GitHub\\ceds-cigar\\LBNL_Simulations\\testpvnum10\\'
Time = list(range(1441))
TotalTimeSteps = len(Time)
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
FilteredOutput_vqvp = np.zeros((TotalTimeSteps,NumberOfNodes))
IntermediateOutput_vqvp= np.zeros((TotalTimeSteps,NumberOfNodes))
Epsilon_vqvp = np.zeros((TotalTimeSteps,NumberOfNodes))

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
V_vqvp = np.zeros((IeeeFeeder,TotalTimeSteps), dtype=np.complex)
S_vqvp = np.zeros((IeeeFeeder,TotalTimeSteps), dtype=np.complex)
IterationCounter_vqvp = np.zeros((IeeeFeeder,TotalTimeSteps))
PowerEachTimeStep_vqvp = np.zeros((IeeeFeeder,3), dtype=np.complex)
SolarGeneration_vqvp = Generation * GenerationScalingFactor
InverterReactivePower = np.zeros(Generation.shape)
InverterRealPower = np.zeros(Generation.shape)
InverterRateOfChangeLimit = 100 
InverterRateOfChangeActivate = 0

#Drop Control Parameters
VQ_start = 1.01
VQ_end = 1.015
VP_start = 1.015
VP_end = 1.02
#VBP is this the config of the control at each time step?
VBP = np.full([IeeeFeeder, 4, TotalTimeSteps], np.nan)
VBP[:,0,0] = VQ_start
VBP[:,1,0] = VQ_end
VBP[:,2,0] = VP_start
VBP[:,3,0] = VP_end
FilteredVoltage = np.zeros(Generation.shape)
FilteredVoltageCalc = np.zeros(Generation.shape)
InverterLPF = 1
ThreshHold_vqvp = 0.25
V0 = np.full((TotalTimeSteps, 1), SlackBusVoltage)
#Adaptive controller parameters
upk = np.zeros(IntermediateOutput_vqvp.shape)
uqk = upk
kq = 100
kp = 100

#Delays                 [  1   2   3*  4   5   6*  7*  8*  9* 10* 11* 12* 13*
Delay_VoltageSampling = [0, 0,  1, 0, 0,  1,  1,  1,  1,  1,  1,  1,  1]
Delay_VBPCurveShift =   [0 ,0 ,2 ,0 ,0 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2] 
print("yay")
##########################################################################################
VBP = np.full([IeeeFeeder, 4, TotalTimeSteps], np.nan)
for i in range(1441):
    VBP[:,0,i] = VQ_start
    VBP[:,1,i] = VQ_end
    VBP[:,2,i] = VP_start
    VBP[:,3,i] = VP_end
##########################################################################################
def voltage_observer(vk, vkm1, psikm1, epsilonkm1, ykm1, f_hp, f_lp, gain, T):
    Vmagk = abs(vk)
    Vmagkm1 = abs(vkm1)
    psik = (Vmagk - Vmagkm1 - (f_hp*T/2-1)*psikm1)/(1+f_hp*T/2)
    epsilonk = gain*(psik**2)
    yk = (T*f_lp*(epsilonk + epsilonkm1) - (T*f_lp - 2)*ykm1)/(2 + T*f_lp)
    return yk, psik, epsilonk

def inverter_VoltVarVoltWatt_model(gammakm1,solar_irr,Vk,Vkm1,VBP,T,lpf,Sbar,pkm1,qkm1,ROC_lim,InverterRateOfChangeActivate,ksim,Delay_VoltageSampling):
    Vmagk = abs(Vk)
    Vmagkm1 = abs(Vkm1) 
    gammakcalc = (T*lpf*(Vmagk + Vmagkm1) - (T*lpf - 2)*gammakm1)/(2 + T*lpf)
    if ksim % Delay_VoltageSampling == 0:
        gammakused = gammakcalc
    else: 
        gammakused = gammakm1
    
    pk = 0
    qk = 0
    c = 0
    q_avail = 0

    if solar_irr < 2500:
        pk = 0
        qk = 0
    elif solar_irr >= 2500:
        if gammakused <= VBP[2]:
            pk = -solar_irr
            q_avail = (Sbar**2 - pk**2)**(1/2)
            if gammakused <= VBP[0]:
                qk = 0
            elif gammakused > VBP[0] and gammakused <= VBP[0]:
                c = q_avail/(VBP[1] - VBP[0])
                qk = c*(gammakused - VBP[0])
            else:
                qk = q_avail       
        elif gammakused > VBP[2] and gammakused < VBP[3]:
            d = -solar_irr/(VBP[3] - VBP[2])
            pk = -(d*(gammakused - VBP[2]) + solar_irr);
            qk = (Sbar**2 - pk**2)**(1/2);      
        elif gammakused >= VBP[3]:
            qk = Sbar
            pk = 0
    return qk,pk,gammakused, gammakcalc, c, q_avail

#########################################################################
class _state(object):
    def __init__(self, PET, V, S, InvReal, InvReact, FV, FVC, Fo, Io, Ep):
        self.PET = PET
        self.V = V
        self.S = S
        self.InvReal = InvReal
        self.InvReact = InvReact
        self.FV = FV
        self.FVC = FVC
        self.Fo = Fo
        self.Io = Io
        self.Ep = Ep
    
    def get_state_agent(self, agent):
        arg = (self.PET[agent], np.array([self.V[agent]]), np.array([self.S[agent]]), np.array([self.InvReal[agent]]), np.array([self.InvReact[agent]]),np.array([self.FV[agent]]),np.array([self.FVC[agent]]))
        return abs(np.concatenate(arg, axis = 0))

class env(object):
    
    ######### reset the env ####################
    def __init__(self):
        self.reset()
    
    def reset(self):
        # a state of env contain: state it is in, terminal or not, in which step it is in
        self.stage = 0
        PET, V, S = self._init_PET_VS()
        InvReal = np.zeros(NumberOfNodes)
        InvReact = np.zeros(NumberOfNodes)
        FV = np.zeros(NumberOfNodes)
        FVC = np.zeros(NumberOfNodes)
        Fo = np.zeros(NumberOfNodes)
        Io = np.zeros(NumberOfNodes)
        Ep = np.zeros(NumberOfNodes)
        self.state = _state(PET, V, S, InvReal, InvReact, FV, FVC, Fo, Io, Ep)
        self.terminal = False
        return self.state.get_state_agent(agent)
    #############################################
    # next state, execute an action #############
    #############################################
    def step(self, action):
        #return next state, reward, terminal or not, precise info
        nextPET = self._cal_next_PET()
        nextV, nextS = self._cal_next_VS(nextPET)
        nextInvReal, nextInvReact, nextFV, nextFVC = self._cal_next_Inv(nextV, action)
        nextFo, nextIo, nextEp = self._cal_next_FIE(nextV)
        nextState = _state(nextPET, nextV, nextS, nextInvReal, nextInvReact, nextFV, nextFVC, nextFo, nextIo, nextEp) 
        
        #update new state, reward and stage
        self.state = nextState
        self.reward = -nextFo
        self.stage += 1
        
        #check if terminal
        if (self.stage == TotalTimeSteps-1):
            self.terminal = True
        
        return self.state.get_state_agent(agent), self.reward, self.terminal
   ################################################     
    def _init_PET_VS(self):
        for knode in LoadList:
            PowerEachTimeStep_vqvp[knode,:] = np.array([ZIP_demand[0,knode,0] - SolarGeneration_vqvp[0,knode],
                                                        ZIP_demand[0,knode,1],
                                                        ZIP_demand[0,knode,2]])
        V, S = FBSfun(matlab_engine, V0[0,0], PowerEachTimeStep_vqvp, Z, FeederMap)
        return PowerEachTimeStep_vqvp, V, S       
    
    def _cal_next_PET(self):
        ksim = self.stage
        currentState = self.state
        for knode in LoadList:
            PowerEachTimeStep_vqvp[knode,:] = np.array([ZIP_demand[ksim+1,knode,0] + currentState.InvReal[knode]
                                                         + 1j*currentState.InvReact[knode], 
                                                        ZIP_demand[ksim+1,knode,1], 
                                                        ZIP_demand[ksim+1,knode,2]])
        return PowerEachTimeStep_vqvp
    
    def _cal_next_VS(self, nextPET):
        ksim = self.stage
        V, S = FBSfun(matlab_engine,V0[ksim+1,0], nextPET, Z,FeederMap)
        return V, S  
    
    def _cal_next_Inv(self, nextV, action):
        ksim = self.stage
        currentState = self.state
        
        InvReal = np.zeros(NumberOfNodes)
        InvReact = np.zeros(NumberOfNodes)
        FV = np.zeros(NumberOfNodes)
        FVC = np.zeros(NumberOfNodes)
        
        for knode in LoadList:
            InvReact[knode], InvReal[knode], FV[knode], FVC[knode], _, _ = inverter_VoltVarVoltWatt_model(
                     currentState.FV[knode], SolarGeneration_vqvp[ksim+1,knode], 
                     abs(nextV[knode]), abs(currentState.V[knode]), 
                     action[knode], TimeStep, InverterLPF, 
                     Sbar[knode], currentState.InvReal[knode], 
                     currentState.InvReact[knode], InverterRateOfChangeLimit, 
                     InverterRateOfChangeActivate, ksim+1, Delay_VoltageSampling[knode])
        return InvReal, InvReact, FV, FVC
    
    # ok
    def _cal_next_FIE(self, nextV):
        currentState = self.state
        Fo = np.zeros(NumberOfNodes)
        Io = np.zeros(NumberOfNodes)
        Ep = np.zeros(NumberOfNodes)
        for knode in LoadList:
            Fo[knode], Io[knode], Ep[knode] = voltage_observer(nextV[knode], currentState.V[knode], 
                                                              currentState.Io[knode], currentState.Ep[knode],
                                                              currentState.Fo[knode], HighPassFilterFrequency,
                                                              LowPassFilterFrequency, Gain_Energy, TimeStep) 
        return Fo, Io, Ep


#create a buffer for training data

from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''     
        batch_size specifies the number of experiences to add 
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least 
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# define actor
class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        #actor network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()
        #create an copy of actor network as target
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        
        #periodically update target network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)+
                                                tf.multiply(self.target_network_params[i], 1. -self.tau))
                                            for i in range(len(self.target_network_params))]
        
        #still not understand
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        self.unnormalized_actor_gradients = tf.gradients(self.scaled_out,self.network_params,-self.action_gradient)
        self.actor_gradients = list(map(lambda x:tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
 
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
        print("Created Actor Network!")
        
    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        #final layer
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval = 0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init = w_init)
        scaled_out = tf.multiply(out, self.action_bound) #action in range (-0.5 to 0.5 -> need to transfer 0.5 and 1)
        return inputs, out, scaled_out
    
    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
    
    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })
    
    def predict_target(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    

# define critic
class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        
        self.inputs, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]
        
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(len(self.network_params)+num_actor_vars):]
        
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
                                             for i in range(len(self.target_network_params))]
        
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        self.action_grads = tf.gradients(self.out, self.action)
        print("Created Critic Network!")
    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)
        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation = 'relu')
        
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out
    
    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict = {
            self.inputs: inputs,
            self.inputs: action,
            self.predicted_q_value: predicted_q_value
        })
    
    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })
    
    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })
    
    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action
        })
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax value", episode_ave_max_q)
    
    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()
    
    return summary_ops, summary_vars

def train(sess, env, args, actor, critic, actor_noise):
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variable_initilizer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)
    actor.update_target_network()
    critic.update_target_network()
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    
    for i in range(int(args['max_episodes'])):
        s = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0
        
        for j in range(int(args['max_episode_len'])):
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
            
            translate = np.array([0.5, 0.5, 1, 1])
            allAction[agent] = a + translate
            s2, r, terminal = env.step(allAction)
            
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
                break
                
#main body

def main(args):
    with tf.Session() as sess:
        silmulation = env()
        tf.set_random_seed(int(args['random_seed']))

        state_dim = 10
        action_dim = 4
        action_bound = 0.5
        # Ensure action bound is symmetric
        print("creating actor!")
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        train(sess, simulation, args, actor, critic, actor_noise)

###############################################
###############################################
args = {}
args['actor_lr'] = 0.0001
args['critic_lr'] = 0.001
args['gamma'] = 0.99
args['tau'] = 0.001
args['buffer_size'] = 10000
args['minibatch_size'] = 64
args['random_seed'] = 1234
args['max-episodes'] = 50000
args['max-episode-len'] = TotalTimeSteps
args['summary_dir'] = os.getcwd()
args['random_seed'] = 1234
#in this chapter, we only want to control 1 agent, other agents have a constant VBP
#agent 
agent = 2
#init action 
allAction = VBP[:,:,0] #need to run simulation

main(args)
