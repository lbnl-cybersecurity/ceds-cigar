# Phasor Measurement Unit - possible input to NN

import numpy as np
import util.signal_processing as signal_processing

class phasor_measurement_unit():
    
    def __init__(self):
        
        pass
    
    def set_timesteps(self, Ts, time, numTimeSteps):
        
        self.Ts = Ts # operation timestep length [s]
        
        self.numTimeSteps = numTimeSteps
                
        self.Vmag = np.zeros((3,numTimeSteps)) # voltage magnitude measurement, 3 phases
        self.Vang = np.zeros((3,numTimeSteps)) # voltage angle measurement, 3 phases

        self.Vcomp = np.zeros((3,numTimeSteps),dtype='complex') # complex voltage measurement, 3 phases
        self.Vreal = np.zeros((3,numTimeSteps)) # real component of voltage measurement, 3 phases
        self.Vimag = np.zeros((3,numTimeSteps)) # imaginary component of voltage measurement, 3 phases
       
        self.kop = 0 # operation timestep index
        self.timeop = np.zeros(numTimeSteps)
        self.Nop = 0
        
    def set_opertime(self, Top, Toff):
        
        self.kop = 0
        
        self.Top = Top
        self.Toff = Toff
        
        self.Tlast = self.Toff
        
    # set the index of the node where the inverter is located
    def set_busidx(self, busidx):
        self.busidx = busidx
        
    # set the name of the node where the inverter is located
    def set_busname(self, busname):
        self.busname = busname

    # voltage observer operation
    def measure_voltage(self, kop, vk):
        pass
        
    # delete extraneous timesteps
    def truncate_time_data(self):
        
        self.Nop = self.kop
        self.timeop = self.timeop[0:self.Nop+1]
        self.Vmag = self.Vmag[:,0:self.Nop+1]
        self.Vang = self.Vang[:,0:self.Nop+1]
        self.Vcomp = self.Vcomp[:,0:self.Nop+1]
        self.Vreal = self.Vreal[:,0:self.Nop+1]
        self.Vimag = self.Vimag[:,0:self.Nop+1]
    