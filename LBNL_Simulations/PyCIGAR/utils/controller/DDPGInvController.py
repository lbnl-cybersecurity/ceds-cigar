# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 09:59:30 2018

@author: Sy-Toan Ngo
"""

import numpy as np
from utils.controller.RLAlgos.ddpg.ddpg import DDPG
import copy 

class DDPGInvController:

	controllerType = 'DDPGInvController' 
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
		self.ddpg = DDPG(sess=sess, logger=self.logger, **rl_kwargs)
		
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
			# do training or store into buffer of DDPG
			state = self.state_processing(self.prevV, self.prevG, self.prevL)
			next_state = self.state_processing(self.V, self.G, self.L)
			self.reward = self.get_reward()
			
			#end of episode
			if self.k == len(self.time)-1:
				self.done = True
			else: 
				self.done = False

			if self.reward: #skip the first length when state is dummy
				self.ddpg.buf.store(state, self.action, self.reward, next_state, self.done)
			

			if self.ddpg.warmUp >= self.ddpg.start_steps:
				self.action = self.sess.run(self.ddpg.pi, feed_dict={self.ddpg.x_ph: next_state.reshape([1]+list(self.ddpg.obs_dim))})
				print("new action:", self.action)

				self.action += self.ddpg.act_noise * np.random.randn(self.ddpg.act_dim[0])
				self.action += self.ddpg.act_shift
				self.action = np.clip(self.action, -self.ddpg.act_limit + self.ddpg.act_shift, self.ddpg.act_limit + self.ddpg.act_shift)
				print("new action:", self.action)
			else:
				self.action = (np.random.randn(self.ddpg.act_dim[0])-1)*self.ddpg.act_limit + self.ddpg.act_shift
				self.ddpg.warmUp += 1

			#update VBP for future timestep
			for i in range(self.k, len(self.time)):
				self.VBP[i] = self.action
				

			if self.reward:
				self.ep_ret += self.reward


			if (self.ddpg.buf.current_size() > self.ddpg.batch_size):
				print("run update")
				self.ddpg.update()
				self.ddpg.dump_logger()	
		
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

