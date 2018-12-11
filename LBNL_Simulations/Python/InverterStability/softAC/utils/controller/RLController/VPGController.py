# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 20:28:36 2018

@author: Daniel Arnold
"""

import numpy as np
import copy

class AdaptiveInvController:

        controllerType = 'AdaptiveInvController' 
        #instance atttributes
        def __init__(self, time, VBP, delayTimer):
                self.time = time        
                self.delT = time[1] - time[0]
                self.VBP = np.zeros((len(time), 4))
                self.up = np.zeros(len(time))
                self.uq = np.zeros(len(time))

                self.delayTimer = delayTimer

                for i in range(len(time)):
                        self.VBP[i] = VBP
                #simulation index
                self.k = 0
                
                self.VBPcounter = 0

        
        
        def act(self, **kwargs):
                nk = kwargs['nk'] 
                device = kwargs['device']
                thresh = kwargs['thresh']
                
                if self.VBPcounter != self.delayTimer:
                        self.VBPcounter += 1
                else: #begin to control things
                        #reset counter
                        yk, vk, vkmdelay = device.get_info(self.delayTimer)
                        
                        self.VBPcounter = 0
                        self.up[self.k] = self.adaptive_control(nk, vk, vkmdelay, self.up[self.k-self.delayTimer+1], thresh, yk)
                        self.uq[self.k] = self.adaptive_control(nk, vk, vkmdelay, self.uq[self.k-self.delayTimer+1], thresh, yk)

                        vbp = np.array([
                                        self.VBP[0][0] - self.uq[self.k],
                                        self.VBP[0][1] + self.uq[self.k], 
                                        self.VBP[0][2] - self.up[self.k], 
                                        self.VBP[0][3] + self.up[self.k]])
                                
                        for i in range(self.k, len(self.time)):
                                self.VBP[i] = copy.deepcopy(vbp)
                                self.up[i] = self.up[self.k]
                                self.uq[i] = self.uq[self.k]
                self.k += 1

        def get_VBP(self):
                return self.VBP[self.k]

        def adaptive_control(self, nk ,vk, vkmdelay, ukmdelay, thresh, yk):
                delay = self.delayTimer

                if (yk > thresh):
                        uk = delay/2* nk * ( vk**2 + vkmdelay**2 ) + ukmdelay
                else:
                        uk = ukmdelay
                return uk
