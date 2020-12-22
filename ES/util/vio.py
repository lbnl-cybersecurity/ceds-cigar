import numpy as np
import util.signal_processing as signal_processing

class voltage_imbalance_observer_manager():

    def __init__(self):
        
        pass


class voltage_imbalance_observer():

    def __init__(self):
        
        pass
    
    def set_timesteps(self, Ts, time, numTimeSteps):
        
        self.Ts = Ts
        
        self.numTimeSteps = numTimeSteps
        
        self.Vimb = np.zeros((4,numTimeSteps))

        self.Vimblpf = np.zeros((4,numTimeSteps))
       
        self.kop = 0
        self.timeop = np.zeros(numTimeSteps)
        self.Nop = 0

    def set_opertime(self, Top, Toff):
        
        self.kop = 0
        
        self.Top = Top
        self.Toff = Toff
        
        self.Tlast = self.Toff
        
        self.flpf = 2.0
        self.wlpf = 2*np.pi*self.flpf
        
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