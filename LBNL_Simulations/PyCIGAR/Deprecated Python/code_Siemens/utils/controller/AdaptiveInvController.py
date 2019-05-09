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
        def __init__(self, time, VBP, delayTimer, device, nk=0.1, threshold=0.25):
                self.time = time        
                self.delT = time[1] - time[0]
                self.VBP = np.zeros((len(time), 4))
                self.initVBP = VBP
                self.up = np.zeros(len(time))
                self.uq = np.zeros(len(time))
                self.delayTimer = 60
                if delayTimer != 0:
                    self.delayTimer = delayTimer
                self.device = device
                self.nk = nk
                self.thresh = threshold
                
                #reset the controller
                self.reset()

                
        
        def reset(self):
                self.k = 0 #count timestep
                
                #counter for VBP change, if VBPCounter == delayTimer, we execute the function to change VBP
                # 0 <= VBPCounter <= delayTimer
                self.VBPCounter = 0 
                #init VBP for all timestep 
                for i in range(len(self.time)):
                        self.VBP[i] = self.initVBP

        def act(self, **kwargs):
                #nk = kwargs['nk'] 
                #thresh = kwargs['thresh']
                
                # if VBPCounter not equal to delayTimer, we do not change the VBP point
                if self.VBPCounter != self.delayTimer:
                        self.VBPCounter += 1
                
                # otherwise, we change VBP point
                else:
                        self.VBPCounter = 0 #reset the VBPCounter
                        yk, vk, vkmdelay = self.device.get_info_adaptive(self.delayTimer) #get yk and vk vkdelay 
                        self.up[self.k] = self.adaptive_control(self.nk, vk, vkmdelay, 
                                                                self.up[self.k-self.delayTimer+1], 
                                                                self.thresh, yk) #calculate up

                        self.uq[self.k] = self.adaptive_control(self.nk, vk, vkmdelay, 
                                                                self.uq[self.k-self.delayTimer+1],
                                                                self.thresh, yk) #calculate uq

                        vbp = np.array([
                                        self.VBP[0][0] - self.uq[self.k],
                                        self.VBP[0][1] + self.uq[self.k], 
                                        self.VBP[0][2] - self.up[self.k], 
                                        self.VBP[0][3] + self.up[self.k]]) #create new VBP
                                
                        #change all VBP from this point onward...
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
