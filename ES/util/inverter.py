# Inverter object class

import numpy as np
import util.signal_processing as signal_processing

class inverter_manager():

    def __init__(Self):

        pass


class inverter():
    
    # Initialize
    def __init__(self):
                
        self.reactive_power = 0 # current reactive power
    
        
        self.VBP = np.array([0.97, 0.99, 1.01, 1.03]) # Volt-VAr curve breakpoints
        
        
        self.wlp = 2*np.pi*2.0 # lowpass filter cutoff frequency        
        
        self.Ts = 1.0 # lowpass filter timestep        
        
        self.Vmeas = [0] # measured voltage
        
        
        self.Vlp = [0] # lowpass filter voltage
        
        
        self.reactive_power = 0 # reactive power
        
        
        self.hackFlag = False # attack status
        
        
        self.VAr_capacity = 200 # VAr capacity        
        
        self.qset = [0] # reactive power setpoint        
        self.qin = [0] # reactive power consumed
    
    # Set timesteps for asynchronous inverter operation
    def set_timesteps(self, Ts, time, numTimeSteps):        
        
        self.Ts = Ts # Simulation timestep        
        
        self.numTimeSteps = numTimeSteps # number of simulation timesteps, based on simulation
        
        self.Vmeas = np.zeros(numTimeSteps) # measured voltage
        self.Vlp = np.zeros(numTimeSteps) # lowpas filtered voltage

        self.reactive_power = np.zeros(numTimeSteps)

        self.pset = np.zeros(numTimeSteps) # active power setpoint
        self.pin = np.zeros(numTimeSteps) # active power consumed

        self.qset = np.zeros(numTimeSteps) # reactive power setpoint
        self.qin = np.zeros(numTimeSteps) # reactive power consumed
        
        self.kop = 0 # inverter operation timestep
        self.timeop = np.zeros(numTimeSteps) # inverter operation timestep simulation times
        self.Nop = 0 # inverter total operation timesteps
        
        self.VBPhist = np.zeros((numTimeSteps,4)) # history of voltage breakpoints
        
    # Set inveter operation timesteps
    def set_opertime(self, Top, Toff):
        
        self.kop = 0 # inverter operation timestep
        
        self.Top = Top # inverter operation timestep length
        self.Toff = Toff # inverter operation time offset
        
        self.Tlast = self.Toff # last inverter operation timestep elapsed
        
    # set the index of the bus where the inverter is located
    def set_busidx(self, busidx):
        self.busidx = busidx
        
    # set the name of the bus where the inverter is located
    def set_busname(self, busname):
        self.busname = busname
    
    # set connection type
    def set_connection(self, conn):
        self.conn = conn
    
    # set phase
    def set_phase(self, phase):
        self.phase = phase
        
    # set the name of the node where the inverter is located
    def set_loadname(self, loadname):
        self.loadname = loadname
    
    # set VAr capacity
    def set_VAr_capacity(self, VAr_capacity):
        self.VAr_capacity = VAr_capacity
    
    # set lowpass filter cutoff frequency
    def set_lowpass_frequency(self, wlp):
        self.wlp = wlp
    
    # set Volt-VAr curve breakpoints
    def set_VBP(self, VBP):
        self.VBP = VBP
    
    # set measured voltage
    def measure_voltage(self, kt, Vmeas):
        self.Vmeas[kt] = Vmeas # measured voltage        
        self.VBPhist[kt,:] = self.VBP # store VBP at current op timestep for history
    
    # lowpass filter voltage measurements
    def lowpass(self, kt):
        # initialize lowpas filter voltage array
        if kt == 0:
            self.Vlp[kt] = self.Vmeas[kt]
        # lowpass filter measured voltage
        else:
            self.Vlp[kt] = (1 - self.wlp*self.Top)*self.Vlp[kt-1] + self.wlp*self.Top*self.Vmeas[kt-1]
            self.Vlp[kt] = 1/(2 + self.wlp*self.Top)*((2 - self.wlp*self.Top)*self.Vlp[kt-1] + self.wlp*self.Top*(self.Vmeas[kt] + self.Vmeas[kt-1]))
    
    # compute reactive power from lowpass filtered voltage and Volt-Var curve
    def compute_reactive_power_output(self, kt, Vcomp):
        # compute percent of VAr capacity to source/sink based on Vlp
        if Vcomp <= self.VBP[0]:
            self.reactive_power[kt] = -100
        elif self.VBP[0] <= Vcomp <= self.VBP[1]:
            self.reactive_power[kt] =100/(self.VBP[1] - self.VBP[0])*(Vcomp - self.VBP[1])
        elif self.VBP[1] <= Vcomp <= self.VBP[2]:
            self.reactive_power[kt] = 0
        elif self.VBP[2] <= Vcomp <= self.VBP[3]:
            self.reactive_power[kt] = 100/(self.VBP[3] - self.VBP[2])*(Vcomp - self.VBP[2])
        elif self.VBP[3] <= Vcomp:
            self.reactive_power[kt] = 100        
        
        self.qset[kt] = 1/100*self.VAr_capacity*self.reactive_power[kt] # reactive power setpoint in kVAr
        self.qin[kt] = self.qset[kt] # reactive power consumed in kVAr
    
    # truncate arrays to total number of operation timesteps for plotting
    def truncate_time_data(self):
            
        self.Nop = self.kop
    
        self.timeop = self.timeop[0:self.Nop+1]

        self.Vmeas = self.Vmeas[0:self.Nop+1]
        self.Vlp = self.Vlp[0:self.Nop+1]

        self.pset = self.pset[0:self.Nop+1]
        self.pin = self.pin[0:self.Nop+1]

        self.reactive_power = self.reactive_power[0:self.Nop+1]
        self.qset = self.qset[0:self.Nop+1]
        self.qin = self.qin[0:self.Nop+1]
        
        self.VBPhist = self.VBPhist[0:self.Nop+1,:]

# # Inverter object class

# class inverter():
    
#     # Initialize
#     def __init__(self):
                
#         self.reactive_power = 0 # current reactive power
    
        
#         self.VBP = np.array([0.97, 0.99, 1.01, 1.03]) # Volt-VAr curve breakpoints
        
        
#         self.wlp = 2*np.pi*2.0 # lowpass filter cutoff frequency        
        
#         self.Ts = 1.0 # lowpass filter timestep        
        
#         self.Vmeas = [0] # measured voltage
        
        
#         self.Vlp = [0] # lowpass filter voltage
        
        
#         self.reactive_power = 0 # reactive power
        
        
#         self.hackFlag = False # attack status
        
        
#         self.VAr_capacity = 200 # VAr capacity        
        
#         self.qset = [0] # reactive power setpoint        
#         self.qin = [0] # reactive power consumed
    
#     # Set timesteps for asynchronous inverter operation
#     def set_timesteps(self, Ts, time, numTimeSteps):        
        
#         self.Ts = Ts # Simulation timestep        
        
#         self.numTimeSteps = numTimeSteps # number of simulation timesteps, based on simulation
        
#         self.Vmeas = np.array([]) # measured voltage
#         self.Vlp = np.array([]) # lowpas filtered voltage

#         self.pset = np.array([0]) # active power setpoint
#         self.pin = np.array([0]) # active power consumed

#         self.qset = np.array([0]) # reactive power setpoint
#         self.qin = np.array([0]) # reactive power consumed
        
#         self.kop = 0 # inverter operation timestep
#         self.timeop = np.array([]) # inverter operation timestep simulation times
#         self.Nop = 0 # inverter total operation timesteps
        
#         self.VBPhist = np.empty(4) # history of voltage breakpoints
        
#     # Set inveter operation timesteps
#     def set_opertime(self, Top, Toff):
        
#         self.kop = 0 # inverter operation timestep
        
#         self.Top = Top # inverter operation timestep length
#         self.Toff = Toff # inverter operation time offset
        
#         self.Tlast = self.Toff # last inverter operation timestep elapsed
        
#     # set the index of the bus where the inverter is located
#     def set_busidx(self, busidx):
#         self.busidx = busidx
        
#     # set the name of the bus where the inverter is located
#     def set_busname(self, busname):
#         self.busname = busname
    
#     # set connection type
#     def set_connection(self, conn):
#         self.conn = conn
    
#     # set phase
#     def set_phase(self, phase):
#         self.phase = phase
        
#     # set the name of the node where the inverter is located
#     def set_loadname(self, loadname):
#         self.loadname = loadname
    
#     # set VAr capacity
#     def set_VAr_capacity(self, VAr_capacity):
#         self.VAr_capacity = VAr_capacity
    
#     # set lowpass filter cutoff frequency
#     def set_lowpass_frequency(self, wlp):
#         self.wlp = wlp
    
#     # set Volt-VAr curve breakpoints
#     def set_VBP(self, VBP):
#         self.VBP = VBP
    
#     # set measured voltage
#     def measure_voltage(self, Vmeas):
#         np.append(self.Vmeas, Vmeas) # measured voltage        
#         np.append(self.VBPhist, self.VBP) # store VBP at current op timestep for history
    
#     # lowpass filter voltage measurements
#     def lowpass(self):
#         # initialize lowpas filter voltage array
#         if kt == 0:
#             np.append(self.Vlp, self.Vmeas)
#         # lowpass filter measured voltage
#         else:
#             np.append(self.Vlp, (1 - self.wlp*self.Top)*self.Vlp[-1] + self.wlp*self.Top*self.Vmeas[-2])
#             np.append(self.Vlp, 1/(2 + self.wlp*self.Top)*((2 - self.wlp*self.Top)*self.Vlp[-1] + self.wlp*self.Top*(self.Vmeas[-1] + self.Vmeas[-2])))

#     # compute active power from lowpass filtered voltage and Volt-Var curve
#     def compute_active_power_output(self, Vcomp):
#         np.append(self.pset, 0)
#         np.append(self.pin, self.pset[-1])
    
#     # compute reactive power from lowpass filtered voltage and Volt-Var curve
#     def compute_reactive_power_output(self, Vcomp):
#         # compute percent of VAr capacity to source/sink based on Vlp
#         if Vcomp <= self.VBP[0]:
#             self.reactive_power[kt] = -100
#         elif self.VBP[0] <= Vcomp <= self.VBP[1]:
#             self.reactive_power[kt] =100/(self.VBP[1] - self.VBP[0])*(Vcomp - self.VBP[1])
#         elif self.VBP[1] <= Vcomp <= self.VBP[2]:
#             self.reactive_power[kt] = 0
#         elif self.VBP[2] <= Vcomp <= self.VBP[3]:
#             self.reactive_power[kt] = 100/(self.VBP[3] - self.VBP[2])*(Vcomp - self.VBP[2])
#         elif self.VBP[3] <= Vcomp:
#             self.reactive_power[kt] = 100        
        
#         np.append(self.qset, 1/100*self.VAr_capacity*self.reactive_power[-1]) # reactive power setpoint in kVAr
#         np.append(self.qin, self.qset[-1]) # reactive power consumed in kVAr
    
#     # truncate arrays to total number of operation timesteps for plotting
#     def truncate_time_data(self):
            
#         self.Nop = self.kop
    
#         self.timeop = self.timeop[0:self.Nop+1]

#         self.Vmeas = self.Vmeas[0:self.Nop+1]
#         self.Vlp = self.Vlp[0:self.Nop+1]

#         self.pset = self.pset[0:self.Nop+1]
#         self.pin = self.pin[0:self.Nop+1]

#         self.reactive_power = self.reactive_power[0:self.Nop+1]
#         self.qset = self.qset[0:self.Nop+1]
#         self.qin = self.qin[0:self.Nop+1]
#         self.VBPhist = self.VBPhist[0:self.Nop+1,:]
        
