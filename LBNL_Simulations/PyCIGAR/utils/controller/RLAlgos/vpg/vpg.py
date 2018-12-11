# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 09:59:30 2018

@author: Sy-Toan Ngo
"""

import numpy as np
import tensorflow as tf
import time
import utils.controller.RLAlgos.vpg.core as core
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class VPGBuffer:
	"""
	A buffer for storing trajectories experienced by a VPG agent interacting
	with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
	for calculating the advantages of state-action pairs.
	"""

	def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
		self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
		self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
		self.adv_buf = np.zeros(size, dtype=np.float32)
		self.rew_buf = np.zeros(size, dtype=np.float32)
		self.ret_buf = np.zeros(size, dtype=np.float32)
		self.val_buf = np.zeros(size, dtype=np.float32)
		self.logp_buf = np.zeros(size, dtype=np.float32)
		self.gamma, self.lam = gamma, lam
		self.ptr, self.path_start_idx, self.max_size = 0, 0, size

	def store(self, obs, act, rew, val, logp):
		"""
		Append one timestep of agent-environment interaction to the buffer.
		"""
		assert self.ptr < self.max_size	 # buffer has to have room so you can store
		self.obs_buf[self.ptr] = obs
		self.act_buf[self.ptr] = act
		self.rew_buf[self.ptr] = rew
		self.val_buf[self.ptr] = val
		self.logp_buf[self.ptr] = logp
		self.ptr += 1

	def finish_path(self, last_val=0):
		"""
		Call this at the end of a trajectory, or when one gets cut off
		by an epoch ending. This looks back in the buffer to where the
		trajectory started, and uses rewards and value estimates from
		the whole trajectory to compute advantage estimates with GAE-Lambda,
		as well as compute the rewards-to-go for each state, to use as
		the targets for the value function.

		The "last_val" argument should be 0 if the trajectory ended
		because the agent reached a terminal state (died), and otherwise
		should be V(s_T), the value function estimated for the last state.
		This allows us to bootstrap the reward-to-go calculation to account
		for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
		"""

		path_slice = slice(self.path_start_idx, self.ptr)
		rews = np.append(self.rew_buf[path_slice], last_val)
		vals = np.append(self.val_buf[path_slice], last_val)
		
		# the next two lines implement GAE-Lambda advantage calculation
		deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
		self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
		
		# the next line computes rewards-to-go, to be targets for the value function
		self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
		
		self.path_start_idx = self.ptr
		#print("ret_buf: ", self.ret_buf)
		#print("val_buf: ", self.val_buf)

	def get(self):
		"""
		Call this at the end of an epoch to get all of the data from
		the buffer, with advantages appropriately normalized (shifted to have
		mean zero and std one). Also, resets some pointers in the buffer.
		"""
		#assert self.ptr == self.max_size	# buffer has to be full before you can get
		self.ptr, self.path_start_idx = 0, 0
		# the next two lines implement the advantage normalization trick
		#adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
		adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
		#print(self.adv_buf, end = "\t")
		#print(adv_mean, end = "\t")
		#print(adv_std, end = "\t")
		self.adv_buf = (self.adv_buf - adv_mean) / adv_std
		return [self.obs_buf, self.act_buf, self.adv_buf, 
				self.ret_buf, self.logp_buf]

"""

Vanilla Policy Gradient

(with GAE-Lambda for advantage estimation)

"""

class VPG:
	def __init__(self, sess, logger, obs_dim=(60,3), act_dim=(4,), actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
		gamma=0.99, pi_lr=3e-4,
		vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
		save_freq=10, buff_size=5):
		"""

		Args:

			actor_critic: A function which takes in placeholder symbols 
				for state, ``x_ph``, and action, ``a_ph``, and returns the main 
				outputs from the agent's Tensorflow computation graph:

				===========  ================  ======================================
				Symbol	   Shape			   Description
				===========  ================  ======================================
				``pi``	   (batch, act_dim)    | Samples actions from policy given 
											   | states.
				``logp``	 (batch,)		   | Gives log probability, according to
											   | the policy, of taking actions ``a_ph``
											   | in states ``x_ph``.
				``logp_pi``  (batch,)		   | Gives log probability, according to
											   | the policy, of the action sampled by
											   | ``pi``.
				``v``		(batch,)		   | Gives the value estimate for states
											   | in ``x_ph``. (Critical: make sure 
											   | to flatten this!)
				===========  ================  ======================================

			ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
				function you provided to VPG.

			seed (int): Seed for random number generators.

			steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
				for the agent and the environment in each epoch.

			epochs (int): Number of epochs of interaction (equivalent to
				number of policy updates) to perform.

			gamma (float): Discount factor. (Always between 0 and 1.)

			pi_lr (float): Learning rate for policy optimizer.

			vf_lr (float): Learning rate for value function optimizer.

			train_v_iters (int): Number of gradient descent steps to take on 
				value function per epoch.

			lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
				close to 1.)

			max_ep_len (int): Maximum length of trajectory / episode / rollout.

			logger_kwargs (dict): Keyword args for EpochLogger.

			save_freq (int): How often (in terms of gap between epochs) to save
				the current policy and value function.
		"""
		self.sess = sess
		self.logger = logger
		#logger.save_config(locals())
		#print(locals())

		seed += 10000 * proc_id() 
		tf.set_random_seed(seed) 
		np.random.seed(seed) 
		
		self.train_v_iters = train_v_iters 
		self.obs_dim = obs_dim 
		self.act_dim = act_dim
		self.buff_size = buff_size

		# Inputs to computation graph
		#####################################change here
		self.x_ph, self.a_ph = core.placeholders(self.obs_dim, self.act_dim)
		#####################################change here
		self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)

		##################################change here
		# Main outputs from computation graph
		self.pi, self.logp, self.logp_pi, self.v = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

		# Need all placeholders in *this* order later (to zip with data from buffer)
		self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

		# Every step, get: action, value, and logprob
		self.get_action_ops = [self.pi, self.v, self.logp_pi]

		# Experience buffer
		self.buf = VPGBuffer(self.obs_dim, self.act_dim, buff_size, gamma, lam)

		# Count variables
		var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
		self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

		# VPG objectives
		self.pi_loss = -tf.reduce_mean(self.logp * self.adv_ph)
		self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

		# Info (useful to watch during learning)
		self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)	  # a sample estimate for KL-divergence, easy to compute
		self.approx_ent = tf.reduce_mean(-self.logp)					   # a sample estimate for entropy, also easy to compute

		# Optimizers
		self.train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)
		self.train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(self.v_loss)

		self.start_time = time.time()

		# Sync params across processes
		#sess.run(sync_all_params())

		# Setup model saving
		#logger.setup_tf_saver(sess, inputs={'x': self.x_ph}, outputs={'pi': pi, 'v': v})

	def dump_logger(self):
        #self.logger.log_tabular('Epoch', epoch)
		self.logger.log_tabular('EpRet', with_min_and_max=True)
		self.logger.log_tabular('EpLen', average_only=True)
		self.logger.log_tabular('VVals', with_min_and_max=True)
		#logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
		self.logger.log_tabular('LossPi', average_only=True)
		self.logger.log_tabular('LossV', average_only=True)
		self.logger.log_tabular('DeltaLossPi', average_only=True)
		self.logger.log_tabular('DeltaLossV', average_only=True)
		self.logger.log_tabular('Entropy', average_only=True)
		self.logger.log_tabular('KL', average_only=True)
		self.logger.log_tabular('Time', time.time()-self.start_time)
		self.logger.dump_tabular()

	def update(self):
		inputs = {k:v for k,v in zip(self.all_phs, self.buf.get())}
		pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

		# Policy gradient step
		self.sess.run(self.train_pi, feed_dict=inputs)

		# Value function learning
		for _ in range(self.train_v_iters):
			self.sess.run(self.train_v, feed_dict=inputs)

		# Log changes from update
		pi_l_new, v_l_new, kl = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl], feed_dict=inputs)
		self.logger.store(LossPi=pi_l_old, LossV=v_l_old, 
					 KL=kl, Entropy=ent, 
					 DeltaLossPi=(pi_l_new - pi_l_old),
					 DeltaLossV=(v_l_new - v_l_old))

	def get_logger(self):
		return self.logger
