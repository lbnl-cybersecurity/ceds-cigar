
import numpy as np
import util.signal_processing as signal_processing

class voltage_oscillation_observer_manager():
    
    def __init__(self):
        
        pass

class voltage_oscillation_observer():
    
    def __init__(self):
        
        pass
    
    def set_timesteps(self, Ts, time, numTimeSteps):
        
        self.Ts = Ts
        
        self.numTimeSteps = numTimeSteps
        
        self.x = np.zeros(numTimeSteps)
        
        self.x = np.zeros(numTimeSteps)
        self.y1 = np.zeros(numTimeSteps)
        self.y2 = np.zeros(numTimeSteps)
        self.y3 = np.zeros(numTimeSteps)
        self.y4 = np.zeros(numTimeSteps)
       
        self.kop = 0
        self.timeop = np.zeros(numTimeSteps)
        self.Nop = 0
        
    def set_opertime(self, Top, Toff):
        
        self.kop = 0
        
        self.Top = Top
        self.Toff = Toff
        
        self.Tlast = self.Toff
        
        self.fosc = 2.0
        self.hp1, temp = signal_processing.butterworth_highpass(4,2*np.pi*1.0*self.fosc)
        self.lp1, temp = signal_processing.butterworth_lowpass(4,2*np.pi*1.0*self.fosc)
        self.bp1num = np.convolve(self.hp1[0, :], self.lp1[0, :])
        self.bp1den = np.convolve(self.hp1[1, :], self.lp1[1, :])
        self.bp1s = np.array([self.bp1num, self.bp1den])
#         self.bp1s = self.hp1
        self.BP1z = signal_processing.c2dbilinear(self.bp1s, self.Top)
        self.lpf2, temp = signal_processing.butterworth_lowpass(2,2*np.pi*self.fosc/2)
        self.LPF2z = signal_processing.c2dbilinear(self.lpf2, self.Top)
        self.nbp1 = self.BP1z.shape[1] - 1
        self.nlpf2 = self.LPF2z.shape[1] - 1
        
    # set the index of the node where the inverter is located
    def set_busidx(self, busidx):
        self.busidx = busidx
        
    # set the name of the node where the inverter is located
    def set_busname(self, busname):
        self.busname = busname

    # set the connection of the node where the inverter is located
    def set_conn(self, conn):
        self.conn = conn

    # set the phase of the node where the inverter is located
    def set_phase(self, phase):
        self.phase = phase

    def observe_voltage(self, kop, vk):
        
#         vk = np.abs(k.node.nodes[node_id]['voltage'][k.time - 1])
#         vkm1 = np.abs(k.node.nodes[node_id]['voltage'][k.time - 2])
#         self.v_meas_k = vk
#         self.v_meas_km1 = vkm1

        self.x[kop] = vk
    
        if kop >= self.BP1z.shape[1]:

            np.sum(-self.BP1z[1,0:-1]*self.y1[kop-self.BP1z.shape[1]+1:kop])
            np.sum(self.BP1z[0,:]*self.x[kop-self.BP1z.shape[1]+1:kop+1])
            
            self.y1[kop] = (1/self.BP1z[1,-1]*(np.sum(-self.BP1z[1,0:-1]*self.y1[kop-self.BP1z.shape[1]+1:kop]) + np.sum(self.BP1z[0,:]*self.x[kop-self.BP1z.shape[1]+1:kop+1])))
            self.y2[kop] = (self.y1[kop]**2)
            self.y3[kop] = (1/self.LPF2z[1,-1]*(np.sum(-self.LPF2z[1,0:-1]*self.y3[kop-self.LPF2z.shape[1]+1:kop]) + np.sum(self.LPF2z[0,:]*self.y2[kop-self.LPF2z.shape[1]+1:kop+1])))
#             self.y4[kop] = np.sqrt(np.abs(self.y3[kop]))
            self.y4[kop] = 1e3*(self.y3[kop])
    
    def truncate_time_data(self):
        
        self.Nop = self.kop
        self.timeop = self.timeop[0:self.Nop+1]
        self.x = self.x[0:self.Nop+1]
        self.y1 = self.y1[0:self.Nop+1]
        self.y2 = self.y2[0:self.Nop+1]
        self.y3 = self.y3[0:self.Nop+1]
        self.y4 = self.y4[0:self.Nop+1]
    