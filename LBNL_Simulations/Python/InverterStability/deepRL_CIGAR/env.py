import pandas as pd
import numpy as np
from math import tan,acos
import os

def FBSfun(V0,loads,Z,B):
    """Summary
    Forward Back Sweep simulation
    Args:
        V0 (float): Description
        loads (TYPE): Description
        Z (TYPE): Description
        B (TYPE): Description
    
    Returns:
        Voltage (V) and Apprent Power (S) complex: V and S
    """
    n=len(Z)
    V=np.zeros(shape=(n,1),dtype=complex)
    s=np.copy(V)
    V[0,0]=V0
    I = np.zeros(shape=(n, 1),dtype=complex)
    I[0,0] = 0

    T=[]
    J=[1]

    for k in range(2,n+1):
        t=np.sum(B[k-1,:])
        if (t==-1):
            T.append(k)
        elif (t>=1):
            J.append(k)
    tol = 0.0001
    itera = 0
    Vtest = 0

    T=np.array(T)-1
    J=np.array(J)-1
    
    while(abs(Vtest-V0) >=tol):
        V[0,0]=V0
        for k in range(0, n - 1):
            idx = np.where(B[k, :] > 0)
            V[idx,0]=V[k,0]-np.multiply(Z[idx,0],I[idx,0])

        for i in range(len(T)-1,-1,-1):
            t=T[i]
            v=np.array([1, abs(V[t,0]), abs(V[t,0])**2])
            s[t,0]=np.dot(loads[t,:],np.transpose(v))
            I[t,0]=np.conj(s[t,0]/V[t,0])
            flag=True
            idx= np.where(B[t,:] == -1)[0]
            while(flag):
                V[idx,0] = V[t,0] + Z[t,0]*I[t,0]
                v=np.array([1, abs(V[idx,0]),abs(V[idx,0])**2])
                s[idx,0]=np.dot(loads[idx,:],np.transpose(v))
                I[idx,0]=np.conj(s[idx,0]/V[idx,0])+I[t,0]
                if (len(np.where(J==idx)[0])==0):
                    t=idx
                    idx=np.where(B[idx,:][0]==-1)[0]
                else:
                    flag=False
        
        for k in range(len(J)-1,0,-1):
            t=J[k]
            v = np.array([1, abs(V[t, 0]), abs(V[t, 0]) ** 2])
            s[t, 0] = np.dot(loads[t, :], np.transpose(v))
            load_current=np.conj(s[t, 0] / V[t, 0])
            idx=np.where(B[t,:] > 0)[0]
            Itot = load_current
            for y in range(0,len(idx)):
                Itot=Itot+I[idx[y],0]
                
            I[t,0]= Itot
            flag= True
            idx = np.where(B[t, :] == -1)[0][0]
            while flag:
                V[idx, 0] = V[t, 0] + Z[t, 0] * I[t, 0]
                v = np.array([1, abs(V[idx, 0]), abs(V[idx, 0]) ** 2])
                s[idx, 0] = np.dot(loads[idx, :], np.transpose(v))
                I[idx, 0] = np.conj(s[idx, 0] / V[idx, 0]) + I[t, 0]
                if (len(np.where(J==idx)[0])==0):
                    t=idx
                    idx=np.where(B[idx,:][0]==-1)[0]
                else:
                    flag=False

        Vtest=V[0,0]
        itera +=1

    V[0,0] = V0
    S=np.multiply(V,np.conj(I))
    
    return np.array(V, dtype=np.complex).squeeze(),np.array(S, dtype=np.complex).squeeze()

def voltage_observer(vk, vkm1, psikm1, epsilonkm1, ykm1, f_hp, f_lp, gain, T):
    """Summary
    run high and low filter to get how good is the voltage at the moment
    """
    Vmagk = abs(vk)
    Vmagkm1 = abs(vkm1)
    psik = (Vmagk - Vmagkm1 - (f_hp*T/2-1)*psikm1)/(1+f_hp*T/2)
    epsilonk = gain*(psik**2)
    yk = (T*f_lp*(epsilonk + epsilonkm1) - (T*f_lp - 2)*ykm1)/(2 + T*f_lp)
    return yk, psik, epsilonk

def inverter_VoltVarVoltWatt_model(gammakm1,solar_irr,Vk,Vkm1,VBP,T,lpf,Sbar,pkm1,qkm1,ROC_lim,InverterRateOfChangeActivate,ksim,Delay_VoltageSampling):
    """Summary
    return the injection of Reactive Power and Active Power with the VBP Curve
    """
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


#state of the entire environment at 1 timestep
class _state(object):

    """Summary
    define a state of the whole system in a single time step, 
    all of the variables below contain the info for all of the nodes in the system
    in that timestep
    Attributes:
        Ep (TYPE): Epsilon_vqvp
        Fo (TYPE): FilteredOutput_vqvp
        FV (TYPE): FilteredVoltage
        FVC (TYPE): FilteredVoltageCalc
        InvReact (TYPE): InverterReactivePower
        InvReal (TYPE): InverterRealPower
        Io (TYPE): IntermediateOutput_vqvp
        PET (TYPE): PowerEachTimeStep_vqvp
        S (TYPE): Apprent Power
        V (TYPE): Voltage
    """
    
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
        """Summary
        get PET, V, S, InvReal, InvReact, FV, FVC of an agent at a specific time step
        Args:
            agent (int): Agent code
        
        Returns:
            np.array: a list of absolute values for that agent - in future, we want to train real part and imagine part separately
        """
        arg = (self.PET[agent], np.array([self.V[agent]]), np.array([self.S[agent]]), np.array([self.InvReal[agent]]), np.array([self.InvReact[agent]]),np.array([self.FV[agent]]),np.array([self.FVC[agent]]))
        return abs(np.concatenate(arg, axis = 0))


class env(object):

    """Summary
    The environment class, 
    Attributes:
        stage (int): timestep
        state (state): state of the whole system at that timestep
        terminal (bool): is it the end of the simulation
    """
    
    def __init__(self):
        """Summary
        initialize new enviroment is to reset the state to zero
        """
        self.reset()
    
    def reset(self):
        """Summary
        reset all variables to zero
        Returns:
            TYPE: Description
        """
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
        
            
        return self.state
    
    def step(self, action):
        """Summary
        
        Args:
            action (np.array): an array of all the actions at all of the nodes
        
        Returns:
            state: next state after taking the action
            terminal: terminal or not
            nextFo: the y_k at the next timestep after taking action
            nextV: the absolute voltage at all nodes at this timestep

        """
        #return next state, reward, terminal or not, precise info
        nextPET = self._cal_next_PET()
        nextV, nextS = self._cal_next_VS(nextPET)
        nextInvReal, nextInvReact, nextFV, nextFVC = self._cal_next_Inv(nextV, action)
        nextFo, nextIo, nextEp = self._cal_next_FIE(nextV)
        nextState = _state(nextPET, nextV, nextS, nextInvReal, nextInvReact, nextFV, nextFVC, nextFo, nextIo, nextEp) 
        
        #update new state, reward and stage
        self.state = nextState
        self.stage += 1
        #check if terminal
        if (self.stage == TotalTimeSteps-1):
            self.terminal = True
            
        return self.state, self.terminal, nextFo, abs(nextV)
   
   	#support functions     
    def _init_PET_VS(self):
        """Summary
        initialize Power at Each Time Step, Voltage and Apprent Power
        """
        for knode in LoadList:
            PowerEachTimeStep_vqvp[knode,:] = np.array([ZIP_demand[0,knode,0] - SolarGeneration_vqvp[0,knode],
                                                        ZIP_demand[0,knode,1],
                                                        ZIP_demand[0,knode,2]])
            
        V, S = FBSfun(V0[0,0], PowerEachTimeStep_vqvp, Z, FeederMap)
        return PowerEachTimeStep_vqvp, V, S       
    
    def _cal_next_PET(self):
        """Summary
        Calculate Power at Each Time Step
        """
        ksim = self.stage
        currentState = self.state
        for knode in LoadList:
            PowerEachTimeStep_vqvp[knode,:] = np.array([ZIP_demand[ksim+1,knode,0] + currentState.InvReal[knode]
                                                         + 1j*currentState.InvReact[knode], 
                                                        ZIP_demand[ksim+1,knode,1], 
                                                        ZIP_demand[ksim+1,knode,2]])
        return PowerEachTimeStep_vqvp
    
    def _cal_next_VS(self, nextPET):
        """Summary
        Calculate Voltage and Apprent Power
        """
        ksim = self.stage
        V, S = FBSfun(V0[ksim+1,0], nextPET, Z,FeederMap)
        return V, S  
    
    def _cal_next_Inv(self, nextV, action):
        """Summary
        Get the injection RealPower and Reactive Power from the curve
        """
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
        """Summary
        run the observer
        """
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
