# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 09:59:30 2018

@author: Sy-Toan Ngo
"""

import numpy as np
from utils.controller.RLAlgos.td3.td3 import TD3
import copy 

class TD3InvController:

	controllerType = 'TD3InvController' 
	#instance atttributes
	def __init__(self, time, VBP, delayTimer, device, sess, logger, rl_kwargs=dict()): #kwargs have args for vpg and logger
		self.logger = logger
		self.sess = sess
		self.time = time
		self.delT = time[1] - time[0]
		self.device = device
		self.delayTimer = delayTimer
		self.initVBP = VBP
		self.reset()
		#init agent
		self.td3 = TD3(sess=sess, logger=self.logger, **rl_kwargs)
		
	# reset internal state of controller
	def reset(self):
		self.V = np.zeros(self.delayTimer)
		self.G = np.zeros(self.delayTimer)
		self.L = np.zeros(self.delayTimer)
		# buffer for delay Timer, last state
		self.prevV = np.zeros(self.delayTimer)
		self.prevG = np.zeros(self.delayTimer)
		self.prevL = np.zeros(self.delayTimer)
		
		self.done = False
		self.ep_ret = 0 #return of episode
		self.ep_len = 0 #
		self.k = 0
		self.VBP = np.zeros((len(self.time),4))
		self.VBPCounter = 0
		#init VBP 
		for i in range(len(self.time)):
			self.VBP[i] = self.initVBP

		self.reward = None
		self.action = self.initVBP
		self.done = False

	#preprocessing VGL for state
	def state_processing(self,V,G,L):
		state = np.array((V,G,L)).T
		from sklearn.preprocessing import scale
		state = scale(state, axis=0)
		return state

	def act(self,V=0,G=0,L=0):
		#accumulate VGL into a sequence
		if self.VBPCounter < self.delayTimer-1 and self.k != len(self.time)-1:
			self.V[self.VBPCounter] = V
			self.G[self.VBPCounter] = G
			self.L[self.VBPCounter] = L
			self.VBPCounter += 1
		
		#when accumulate enough...
		elif self.VBPCounter == self.delayTimer-1 or self.k == len(self.time)-1: #cutoff delayTimer or end of episode
			# do training or store into buffer of TD3
			state = self.state_processing(self.prevV, self.prevG, self.prevL)
			next_state = self.state_processing(self.V, self.G, self.L)
			self.reward = self.get_reward()
			
			#end of episode
			if self.k == len(self.time)-1:
				self.done = True
			else: 
				self.done = False

			if self.reward: #skip the first length when state is dummy
				self.td3.buf.store(state, self.action - self.td3.act_shift, self.reward, next_state, self.done)
			

			if self.td3.warmUp >= self.td3.start_steps:
				self.action = self.sess.run(self.td3.pi, feed_dict={self.td3.x_ph: next_state.reshape([1]+list(self.td3.obs_dim))})
				print("new action:", self.action)

				self.action += self.td3.act_noise * np.random.randn(self.td3.act_dim[0])
				self.action += self.td3.act_shift
				self.action = np.clip(self.action, -self.td3.act_limit + self.td3.act_shift, self.td3.act_limit + self.td3.act_shift)
				print("new action:", self.action)
			else:
				self.action = (np.random.randn(self.td3.act_dim[0])-1)*self.td3.act_limit + self.td3.act_shift
				self.td3.warmUp += 1

			#update VBP for future timestep
			for i in range(self.k, len(self.time)):
				self.VBP[i] = self.action
				

			if self.reward:
				self.ep_ret += self.reward


			if (self.td3.buf.current_size() > self.td3.batch_size):
				print("run update")
				self.td3.update()
				self.td3.dump_logger()	
		
			#reset VBPCounter
			self.VBPCounter = 0 #reset VBP counter
			self.prevV = copy.deepcopy(self.V) # store last state
			self.prevG = copy.deepcopy(self.G) # store last state
			self.prevL = copy.deepcopy(self.L) # store last state
		 
		
		self.k += 1

	def get_reward(self):
		rawy = self.device.get_info_rl(self.delayTimer)
		#print(rawy)
		y = -np.sum(rawy**2)/1000
		return y

	def get_VBP(self):
		return self.VBP[self.k,:]

