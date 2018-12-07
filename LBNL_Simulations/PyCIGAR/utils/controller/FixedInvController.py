# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 20:28:36 2018

@author: Daniel Arnold
"""

import numpy as np

class FixedInvController:

	controllerType = 'FixedInvController' 
	#instance atttributes
	def __init__(self, time, VBP, device):
		self.device = device
		self.time = time	
		self.delT = time[1]- time[0]
		self.VBP = np.zeros([len(time), 4])
		for i in range(len(time)):
			self.VBP[i] = VBP
		#simulation index
		self.reset()

	def act(self, **kwargs):
		self.k += 1

	def get_VBP(self):
		return self.VBP[self.k-1] 

	def reset(self):
		self.k = 0