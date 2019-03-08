# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 20:28:36 2018

@author: Daniel Arnold
"""

import numpy as np

class Inverter:
    
    # instance attributes
    def __init__(self, time, lpf_meas, lpf_output):
        self.time = time
        self.lpf_meas = lpf_meas
        self.lpf_output = lpf_output
        
        self.delT = time[1] - time[0]
        
        self.v_meas = np.zeros(len(time))
        self.v_lpf = np.zeros(len(time))
        self.p_set = np.zeros(len(time))
        self.q_set = np.zeros(len(time))
        self.p_out = np.zeros(len(time))
        self.q_out = np.zeros(len(time))
        self.VBP = np.zeros(4)
        
        #observer
        self.psi = np.zeros(len(time))
        self.epsilon = np.zeros(len(time))
        self.y = np.zeros(len(time))
        #simulation index
        self.k = 0
        
    def get_V_lpf(self):
        return (self.v_lpf)
        
    def step(self, v, solar_irr, solar_minval, Sbar, VBP):
        
        #update volt var and volt watt parameters
        self.VBP = VBP
        #record voltage magnitude measurement
        self.v_meas[self.k] = np.abs(v)
        
        T = self.delT
        lpf_m = self.lpf_meas
        lpf_o = self.lpf_output
        
        pk = 0
        qk = 0
        
        if(self.k > 0):
            
            #compute v_lpf (lowpass filter)
            #gammakcalc = (T*lpf*(Vmagk + Vmagkm1) - (T*lpf - 2)*gammakm1)/(2 + T*lpf)
            self.v_lpf[self.k] = \
            (T * lpf_m * (self.v_meas[self.k] + self.v_meas[self.k-1]) \
             - (T * lpf_m - 2) * (self.v_lpf[self.k-1]) ) / (2 + T * lpf_m)
            
            #compute p_set and q_set
            if (solar_irr >= solar_minval):
                if (self.v_lpf[self.k] <= VBP[2]):
                    #no curtailment
                    pk = -solar_irr
                    q_avail = (Sbar**2 - pk**2)**(1/2)
                    
                    #determine VAR support
                    if (self.v_lpf[self.k] <= VBP[0]):
                        #no VAR support
                        pass
                    elif (self.v_lpf[self.k] > VBP[0] \
                          and self.v_lpf[self.k] <= VBP[1]):
                        #partial VAR suport
                        c = q_avail/(VBP[1] - VBP[0])
                        qk = c*(self.v_lpf[self.k] - VBP[0])
                    else:
                        #full VAR support
                        qk = q_avail
                elif (self.v_lpf[self.k] > VBP[2] \
                      and self.v_lpf[self.k] < VBP[3]):
                    #partial real power curtailment
                    d = -solar_irr / (VBP[3] - VBP[2])
                    pk = -(d * (self.v_lpf[self.k] - VBP[2]) + solar_irr)
                    qk = (Sbar**2 - pk**2)**(1/2)
                elif (self.v_lpf[self.k] >= VBP[3]):
                    #full real power curtailment for VAR support
                    qk = Sbar
                    pk = 0
                    
            self.p_set[self.k] = pk
            self.q_set[self.k] = qk
        
            #compute p_out and q_out
            self.p_out[self.k] = \
            (T * lpf_o * (self.p_set[self.k] + self.p_set[self.k-1]) \
             - (T * lpf_o - 2) * (self.p_out[self.k-1]) ) / (2 + T * lpf_o)
            
            self.q_out[self.k] = \
            (T * lpf_o * (self.q_set[self.k] + self.q_set[self.k-1]) \
             - (T * lpf_o - 2) * (self.q_out[self.k-1]) ) / (2 + T * lpf_o)
        
        #advance simulation index
        self.currentk = self.k * 1
        
        if self.k > 0:
            self.observe()

        self.k = self.k + 1
        return (pk, qk)

    def observe(self, f_hp=1, f_lp=0.1, gain=1e5):
        
        T = self.delT
        vk = self.v_meas[self.currentk]
        vkm1 = self.v_meas[self.currentk-1]
        psikm1 = self.psi[self.currentk-1]
        epsilonkm1 = self.epsilon[self.currentk-1]
        ykm1 = self.y[self.currentk-1]

        self.psi[self.currentk] = psik = (vk - vkm1 - (f_hp*T/2-1)*psikm1)/(1+f_hp*T/2)
        self.epsilon[self.currentk] = epsilonk = gain*(psik**2)
        self.y[self.currentk] = yk = (T*f_lp*(epsilonk + epsilonkm1) - (T*f_lp - 2)*ykm1)/(2 + T*f_lp)

        #return yk, psik, epsilonk

    def get_info(self, length):
        return self.y[self.currentk], self.psi[self.currentk], self.psi[self.currentk-length+1] 
    