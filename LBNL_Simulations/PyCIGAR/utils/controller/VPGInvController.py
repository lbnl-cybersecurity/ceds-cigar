# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 09:59:30 2018

@author: Sy-Toan Ngo
"""

import numpy as np
from utils.controller.RLAlgos.vpg.vpg import VPG
import copy 

class VPGInvController:

	controllerType = 'VPGInvController' 
	#instance atttributes

	def __init__(self, time, VBP, delayTimer, device, sess, logger_kwargs=dict(), rl_kwargs=dict()): #kwargs have args for vpg and logger
		self.logger_kwargs = logger_kwargs

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

		self.vpg = VPG(sess=sess, logger_kwargs=self.logger_kwargs, **rl_kwargs)

		
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
		self.v_t = None
		self.logp_t = None
		self.done = False

	#preprocessing VGL for state
	def state_processing(self,V,G,L):
		state = np.array((V,G,L)).T
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
			# do training or store into buffer of VPG
			state = self.state_processing(self.prevV, self.prevG, self.prevL)
			
			if self.reward and self.v_t and self.logp_t: #skip the first length when state is dummy
				self.vpg.buf.store(state, self.action, self.reward, self.v_t, self.logp_t)
				self.vpg.logger.store(VVals=self.v_t)
			
			next_state = self.state_processing(self.V, self.G, self.L)
			self.reward = self.get_reward()
			
			#end of episode
			if self.k == len(self.time)-1:
				self.done = True
			else: 
				self.done = False

			self.action, self.v_t, self.logp_t = self.sess.run(self.vpg.get_action_ops, feed_dict={self.vpg.x_ph: next_state.reshape([1]+list(self.vpg.obs_dim))})
			#update VBP for future timestep
			for i in range(self.k, len(self.time)):
				self.VBP[i] = self.action
				

			if self.reward:
				self.ep_ret += self.reward
			self.ep_len += 1

			if self.done or (self.ep_len == self.vpg.buff_size):
				if not(self.done):
					print('Warning: trajectory cut off by epoch at %d steps.'%(self.ep_len*self.delayTimer))
				# if trajectory didn't reach terminal state, bootstrap value target
				last_val = self.reward if self.done else self.sess.run(self.vpg.v, feed_dict={self.vpg.x_ph: next_state.reshape([1]+list(self.vpg.obs_dim))})
				self.vpg.buf.finish_path(last_val)
				
				self.vpg.logger.store(EpRet=self.ep_ret, EpLen=self.ep_len)
				# reset ep_len
				self.ep_len = 0
							
				self.vpg.update()
				self.vpg.dump_logger()	
		
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

