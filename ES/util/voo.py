
import numpy as np
import util.signal_processing as signal_processing

class voltage_oscillation_observer_manager():
    
    def __init__(self, time, Ts, jsondata):

        self.voolist = []

        self.time = time
        self.Ts = Ts

        self.jsondata = jsondata
        
        
    def parse_json(self):

        self.voosjson = self.jsondata['voos']

        for k1 in range(len(self.voosjson)):

            tempvoo = voltage_oscillation_observer()

            Toff = 1/100*np.floor(10*np.random.rand())

            tempvoo.set_timesteps(self.Ts,self.time,len(self.time))
            # tempesc.set_opertime(self.escsjson[k1]['Top'],self.escsjson[k1]['Toff'])
            tempvoo.set_opertime(self.voosjson[k1]['Top'],Toff,self.voosjson[k1]['fvoo'])

            tempvoo.set_voo_frequency(self.voosjson[k1]['fvoo'])
            
            tempvoo.set_busname(str(self.voosjson[k1]['bus']))
            tempvoo.set_conn(self.voosjson[k1]['conn'])
            # tempvoo.set_phase(self.voosjson[k1]['phase'])

            tempvoo.set_phase(np.asarray(self.voosjson[k1]['phase'].split('.'), dtype=int))
            print(np.asarray(self.voosjson[k1]['phase'].split('.'), dtype=int))

            self.voolist.append(tempvoo)

        return self.voolist

class voltage_oscillation_observer():
    
    def __init__(self):
        
        pass
    
    def set_timesteps(self, Ts, time, numTimeSteps):
        
        self.Ts = Ts # operation timestep length [s]
        
        self.numTimeSteps = numTimeSteps
        
        self.x = np.zeros(numTimeSteps)
        
        self.x = np.zeros(numTimeSteps) # voltage magnitude measurement
        self.y1 = np.zeros(numTimeSteps) # output of highpass filter
        self.y2 = np.zeros(numTimeSteps) # output of square function
        self.y3 = np.zeros(numTimeSteps) # output of lowpass filter
        self.y4 = np.zeros(numTimeSteps) # output of rectifier gain
       
        self.kop = 0 # operation timestep index
        self.timeop = np.zeros(numTimeSteps)
        self.Nop = 0
        
    def set_opertime(self, Top, Toff, fosc):
        
        self.kop = 0
        
        self.Top = Top
        self.Toff = Toff
        
        self.Tlast = self.Toff
        
        # Frequency on which the observer concentrates
        # Oscillatory content both below and above this value is attenuated
        self.fosc = 2.0 # highpass filter and lowpass filter cutoff frequency [Hz]
        self.hp1, temp = signal_processing.butterworth_highpass(4,2*np.pi*1.0*self.fosc) # highpass filter
        self.lp1, temp = signal_processing.butterworth_lowpass(4,2*np.pi*1.0*self.fosc) # lowpass filter
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

    def set_voo_frequency(self, fvoo):

        # Frequency on which the observer concentrates
        # Oscillatory content both below and above this value is attenuated
        self.fosc = fvoo # highpass filter and lowpass filter cutoff frequency [Hz]
        self.hp1, temp = signal_processing.butterworth_highpass(4,2*np.pi*1.0*self.fosc) # highpass filter
        self.lp1, temp = signal_processing.butterworth_lowpass(4,2*np.pi*1.0*self.fosc) # lowpass filter
        self.bp1num = np.convolve(self.hp1[0, :], self.lp1[0, :])
        self.bp1den = np.convolve(self.hp1[1, :], self.lp1[1, :])
        self.bp1s = np.array([self.bp1num, self.bp1den])
#         self.bp1s = self.hp1
        self.BP1z = signal_processing.c2dbilinear(self.bp1s, self.Top)
        self.lpf2, temp = signal_processing.butterworth_lowpass(2,2*np.pi*self.fosc/2)
        self.LPF2z = signal_processing.c2dbilinear(self.lpf2, self.Top)
        self.nbp1 = self.BP1z.shape[1] - 1
        self.nlpf2 = self.LPF2z.shape[1] - 1

    # voltage observer operation
    def observe_voltage(self, kop, vk):
        
#         vk = np.abs(k.node.nodes[node_id]['voltage'][k.time - 1])
#         vkm1 = np.abs(k.node.nodes[node_id]['voltage'][k.time - 2])
#         self.v_meas_k = vk
#         self.v_meas_km1 = vkm1

        # voltage measurement input
        self.x[kop] = vk
    
        if kop >= self.BP1z.shape[1]:

            np.sum(-self.BP1z[1,0:-1]*self.y1[kop-self.BP1z.shape[1]+1:kop])
            np.sum(self.BP1z[0,:]*self.x[kop-self.BP1z.shape[1]+1:kop+1])
            
            # lowpass filter
            self.y1[kop] = (1/self.BP1z[1,-1]*(np.sum(-self.BP1z[1,0:-1]*self.y1[kop-self.BP1z.shape[1]+1:kop]) + np.sum(self.BP1z[0,:]*self.x[kop-self.BP1z.shape[1]+1:kop+1])))
            # square output of LPF
            self.y2[kop] = (self.y1[kop]**2)
            # highpass filter
            self.y3[kop] = (1/self.LPF2z[1,-1]*(np.sum(-self.LPF2z[1,0:-1]*self.y3[kop-self.LPF2z.shape[1]+1:kop]) + np.sum(self.LPF2z[0,:]*self.y2[kop-self.LPF2z.shape[1]+1:kop+1])))
#             self.y4[kop] = np.sqrt(np.abs(self.y3[kop]))
            # apply gain
            self.y4[kop] = 1e3*(self.y3[kop])
    
    # delete extraneous timesteps
    def truncate_time_data(self):
        
        self.Nop = self.kop
        self.timeop = self.timeop[0:self.Nop+1]
        self.x = self.x[0:self.Nop+1]
        self.y1 = self.y1[0:self.Nop+1]
        self.y2 = self.y2[0:self.Nop+1]
        self.y3 = self.y3[0:self.Nop+1]
        self.y4 = self.y4[0:self.Nop+1]
    