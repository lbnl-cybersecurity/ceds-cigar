# ESC object class

import numpy as np
import signal_processing as signal_processing

class esc_manager():

    def __init__(self):

        pass

class esc():
    
    # Initialize
    def __init__(self):
    
        # init for signal processing on Voltage
        self.Top = 1
        self.fosc = 0.15

#         self.hp1, temp = signal_processing.butterworth_highpass(2,2*np.pi*0.5*self.fosc)
#         self.lp1, temp = signal_processing.butterworth_lowpass(4,2*np.pi*2*self.fosc)
#         self.bp1num = np.convolve(self.hp1[0, :], self.lp1[0, :])
#         self.bp1den = np.convolve(self.hp1[1, :], self.lp1[1, :])
#         self.bp1s = np.array([self.bp1num, self.bp1den])
# #         self.bp1s = self.hp1
#         self.BP1z = signal_processing.c2dbilinear(self.bp1s, self.Top)
#         self.lpf2, temp = signal_processing.butterworth_lowpass(2,2*np.pi*self.fosc/2)
#         self.LPF2z = signal_processing.c2dbilinear(self.lpf2, self.Top)
#         self.nbp1 = self.BP1z.shape[1] - 1
#         self.nlpf2 = self.LPF2z.shape[1] - 1

#         self.x = deque([0]*(len(self.BP1z[0, :]) + step_buffer*2), maxlen=(len(self.BP1z[0, :]) + step_buffer*2))
#         self.y1 = deque([0]*len(self.BP1z[1, 0:-1]), maxlen=len(self.BP1z[1, 0:-1]))
#         self.y2 = deque([0]*len(self.LPF2z[0, :]), maxlen=len(self.LPF2z[0, :]))
#         self.y3 = deque([0]*len(self.LPF2z[1, 0:-1]), maxlen=len(self.LPF2z[1, 0:-1]))

        self.fes = 0.1 # ES probe frequency [Hz]
        self.wes = 2*np.pi*self.fes # ES probe angular frequency [rad/s]
        self.aes = 20 # ES probe amplitude [kW or kVAr]
        self.wh = self.wes/10 # ES highpass filter frequency [rad/s]
        self.wl = self.wes/10 # ES lowpass filter frequency [rad/s]
        self.kes = 5e5 # ES integrator gain

        self.pmin = -100 # minimum active power
        self.pmax = 100 # maximum active power
        self.qmin = -100 # minimum reactive power
        self.qmax = 100 # maximum reactive power

        self.smax = 100 # maximum apaprent power
        
    # Set timesteps and initialize vectors for asynchronous ESC operation
    def set_timesteps(self, Ts, time, numTimeSteps):
        
        self.Ts = Ts # simulation timestep
        
        self.numTimeSteps = numTimeSteps # number of simulation timesteps
                
        self.psi = np.zeros(numTimeSteps) # objective function
        
        self.rhop = np.zeros(numTimeSteps) # highpass filtered objective function for active power probe
        self.epsp = np.zeros(numTimeSteps) # lowpass filtered objective function for active power probe 
        self.sigmap = np.zeros(numTimeSteps) # demodulated value for active power probe
        self.xip = np.zeros(numTimeSteps) # gradient estimate, lowpass filtered demodulated value for active power probe

        self.phat = np.zeros(numTimeSteps) # active power setpoint
        self.p = np.zeros(numTimeSteps) # active power control

        self.rhoq = np.zeros(numTimeSteps) # highpass filtered objective function for reactive power probe
        self.epsq = np.zeros(numTimeSteps) # lowpass filtered objective function for reactive power probe
        self.sigmaq = np.zeros(numTimeSteps) # demodulated value for reactive power probe
        self.xiq = np.zeros(numTimeSteps) # gradient estimate, lowpass filtered demodulated value for reactive power probe

        self.qhat = np.zeros(numTimeSteps) # reactive power setpoint
        self.q = np.zeros(numTimeSteps) # reactive power control
       
        self.kop = 0 # ES operation timestep
        self.timeop = np.zeros(numTimeSteps) # ES operation time array
        self.Nop = 0 # ES total operation timesteps
        
    def set_opertime(self, Top, Toff):
        
        self.kop = 0 # ES operation timestep index
        
        self.Top = Top # ES operation timestep length
        self.Toff = Toff # ES operation time offset
        
        self.Tlast = self.Toff # ES last operation timestep
        
    def set_esc_params(self, fes, aes, kes):

        self.fes = fes # ES probe frequency [Hz]
        self.wes = 2*np.pi*self.fes # ES probe angular frequency [rad/sec]
        self.aes = aes # ES probe amplitude [kW or kVAr]
        self.wh = self.wes/10 # ES highpass filter cutoff frequency [rad/sec]
        self.wl = self.wes/10 # ES lowpass filter cutoff frequency [rad/sec]
        self.kes = kes # ES integrator gain

    def set_esc_limits(self, pmin, pmax, qmin, qmax, smax):

        self.pmin = pmin # minimum active power output
        self.pmax = pmax # maximum active power output
        self.qmin = qmin # minimum reactive power output
        self.qmax = qmax # maximum reactive power output
        self.smax = smax # maximum apparent power output
        
    # set the index of the node where the inverter is located
    def set_busidx(self, busidx):
        self.busidx = busidx
        
    # set the name of the node where the inverter is located
    def set_busname(self, busname):
        self.busname = busname

    # set connection type
    def set_connection(self, conn):
        self.conn = conn
        
    # set phase(s)
    def set_phase(self, phase):
        self.phase = phase
        
    # set the name of the opendss load
    def set_loadname(self, loadname):
        self.loadname = loadname
    
    # not used right now
    def observer(self, kt, vk):
        
#         vk = np.abs(k.node.nodes[node_id]['voltage'][k.time - 1])
#         vkm1 = np.abs(k.node.nodes[node_id]['voltage'][k.time - 2])
#         self.v_meas_k = vk
#         self.v_meas_km1 = vkm1
        self.x[kt] = vk
    
        if kt >= self.BP1z.shape[1]:

            np.sum(-self.BP1z[1,0:-1]*self.y1[kt-self.BP1z.shape[1]+1:kt])
            np.sum(self.BP1z[0,:]*self.x[kt-self.BP1z.shape[1]+1:kt+1])
            
            self.y1[kt] = (1/self.BP1z[1,-1]*(np.sum(-self.BP1z[1,0:-1]*self.y1[kt-self.BP1z.shape[1]+1:kt]) + np.sum(self.BP1z[0,:]*self.x[kt-self.BP1z.shape[1]+1:kt+1])))
            self.y2[kt] = (self.y1[kt]**2)
            self.y3[kt] = (1/self.LPF2z[1,-1]*(np.sum(-self.LPF2z[1,0:-1]*self.y3[kt-self.LPF2z.shape[1]+1:kt]) + np.sum(self.LPF2z[0,:]*self.y2[kt-self.LPF2z.shape[1]+1:kt+1])))
#             self.y4[kt] = np.sqrt(np.abs(self.y3[kt]))
            self.y4[kt] = 1e3*(self.y3[kt])
    
    # receive and store objective function value
    def receive_objective(self, kop, psik):
        
        self.psi[kop] = psik
    
    # ES operation at each ES timestep
    def esc_function(self, kop, timevalk, timevalkm1):

        # at first timestep do nothing
        # otherwise, ES 
        if kop >= 1:

            # highpass filter objective function
            self.rhop[kop] = (1 - self.Top*self.wh)*self.rhop[kop-1] + self.psi[kop] - self.psi[kop-1]
            self.rhop[kop] = 1/(2 + self.Top*self.wh)*((2 - self.Top*self.wh)*self.rhop[kop-1] + 2*(self.psi[kop] - self.psi[kop-1]))

            # lowpass filtere objective function - subtract rho fom objective
            self.epsp[kop] = self.psi[kop] - self.rhop[kop]

            # demodulate - multiply by probe divide by amplitude
            self.sigmap[kop] = 2/self.aes*np.cos(self.wes*timevalkm1)*self.rhop[kop]

            # lowpass filter to obtain gradient estimate
            self.xip[kop] = (1 - self.Top*self.wl)*self.xip[kop-1] + self.Top*self.wl*self.sigmap[kop-1]
            self.xip[kop] = 1/(2 + self.Top*self.wl)*((2 - self.Top*self.wl)*self.xip[kop-1] + self.Top*self.wl*(self.sigmap[kop] + self.sigmap[kop-1]))

            # only integrate gradient estimate to update setpoint if objective function is above threshold
            # otherwise, keep setpoint constant
            if np.abs(self.psi[kop]) >= 1e-99:
                self.phat[kop] = self.phat[kop-1] - 0.1*1*self.Top*self.kes*self.xip[kop-1]
                self.phat[kop] = self.phat[kop-1] - 0.1*1/2*self.Top*self.kes*(self.xip[kop] + self.xip[kop-1])
            else:
                self.phat[kop] = self.phat[kop-1]
                self.phat[kop] = self.phat[kop-1]

            # self.phat[kop] = 0

            # self.phat[kop] = 0

            # # rectify setpoint
            # if self.phat[kop] <= self.pmin + self.aes:
            #     self.phat[kop] = self.pmin + self.aes
            # if self.phat[kop] >= self.pmax - self.aes:
            #     self.phat[kop] = self.pmax - self.aes

            # # modulate - add probe to setpoint
            # self.p[kop] = self.phat[kop] + self.aes*np.cos(self.wes*timevalk)

        # at first timestep do nothing
        # otherwise, ES for reactive power
        if kop >= 1:

            # highpass filter objective function
            self.rhoq[kop] = (1 - self.Top*self.wh)*self.rhoq[kop-1] + self.psi[kop] - self.psi[kop-1]
            self.rhoq[kop] = 1/(2 + self.Top*self.wh)*((2 - self.Top*self.wh)*self.rhoq[kop-1] + 2*(self.psi[kop] - self.psi[kop-1]))

            # lowpass filtere objective function - subtract rho fom objective
            self.epsq[kop] = self.psi[kop] - self.rhoq[kop]

            # demodulate - multiply by probe divide by amplitude
            self.sigmaq[kop] = 2/self.aes*np.sin(self.wes*timevalkm1)*self.rhoq[kop]

            # lowpass filter to obtain gradient estimate
            self.xiq[kop] = (1 - self.Top*self.wl)*self.xiq[kop-1] + self.Top*self.wl*self.sigmaq[kop-1]
            self.xiq[kop] = 1/(2 + self.Top*self.wl)*((2 - self.Top*self.wl)*self.xiq[kop-1] + self.Top*self.wl*(self.sigmaq[kop] + self.sigmaq[kop-1]))

            # only integrate gradient estimate to update setpoint if objective function is above threshold
            # otherwise, keep setpoint constant
            if np.abs(self.psi[kop]) >= 1e-99:
                self.qhat[kop] = self.qhat[kop-1] - 1*self.Top*self.kes*self.xiq[kop-1]
                self.qhat[kop] = self.qhat[kop-1] - 1/2*self.Top*self.kes*(self.xiq[kop] + self.xiq[kop-1])
            else:
                self.qhat[kop] = self.qhat[kop-1]
                self.qhat[kop] = self.qhat[kop-1]

            # rectify setpoint

            # rectify active power setpoint for min/max value
            if self.phat[kop] <= self.pmin + self.aes:
                self.phat[kop] = self.pmin + self.aes
            if self.phat[kop] >= self.pmax - self.aes:
                self.phat[kop] = self.pmax - self.aes

            # rectify reactive power setpoint for min/max value
            if self.qhat[kop] <= self.qmin + self.aes:
                self.qhat[kop] = self.qmin + self.aes
            if self.qhat[kop] >= self.qmax - self.aes:
                self.qhat[kop] = self.qmax - self.aes
            
            # rectify setpoint for max apparent power value
            # if self.phat[kop]**2 + self.qhat[kop]**2 >= (self.smax - self.aes)**2:
            #     gammatemp = (self.smax - self.aes)/(self.phat[kop]**2 + self.qhat[kop]**2)**0.5

            #     self.phat[kop] = self.phat[kop]*gammatemp
            #     self.qhat[kop] = self.qhat[kop]*gammatemp

            # modulate - add probe to setpoint
            self.p[kop] = self.phat[kop] + self.aes*np.cos(self.wes*timevalk)

            # modulate - add probe to setpoint
            self.q[kop] = self.qhat[kop] + self.aes*np.sin(self.wes*timevalk)
            
    def truncate_time_data(self):
        
        self.Nop = self.kop
    
        self.timeop = self.timeop[0:self.Nop+1]

        self.psi = self.psi[0:self.Nop+1]

        self.rhop = self.rhop[0:self.Nop+1]
        self.epsp = self.epsp[0:self.Nop+1]
        self.sigmap = self.sigmap[0:self.Nop+1]
        self.xip = self.xip[0:self.Nop+1]

        self.phat = self.phat[0:self.Nop+1]
        self.p = self.p[0:self.Nop+1]

        self.rhoq = self.rhoq[0:self.Nop+1]
        self.epsq = self.epsq[0:self.Nop+1]
        self.sigmaq = self.sigmaq[0:self.Nop+1]
        self.xiq = self.xiq[0:self.Nop+1]

        self.qhat = self.qhat[0:self.Nop+1]
        self.q = self.q[0:self.Nop+1]
        