# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 09:59:30 2018

@author: Sy-Toan Ngo
"""

import numpy as np
import tensorflow as tf
import time
import utils.controller.RLAlgos.trpo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

EPS = 1e-8

class TRPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32) for k,v in info_shapes.items()}
        self.sorted_info_keys = core.keys_as_sorted_list(self.info_bufs)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, info):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.sorted_info_keys):
            self.info_bufs[k][self.ptr] = info[i]
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

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        #assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        #adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf] + core.values_as_sorted_list(self.info_bufs)


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
class TRPO:

    def __init__(self, sess, logger, obs_dim=(60,3), act_dim=(4,), actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
            gamma=0.99, delta=0.01, clip_ratio=0.2, pi_lr=3e-4,
            vf_lr=1e-3, damping_coeff=0.1, cg_iters=10, backtrack_iters=10, 
            backtrack_coeff=0.8, train_v_iters=80, lam=0.97,
            target_kl=0.01, save_freq=10, buff_size=5, algo='npg'):
        """

        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: A function which takes in placeholder symbols 
                for state, ``x_ph``, and action, ``a_ph``, and returns the main 
                outputs from the agent's Tensorflow computation graph:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                               | states.
                ``logp``     (batch,)          | Gives log probability, according to
                                               | the policy, of taking actions ``a_ph``
                                               | in states ``x_ph``.
                ``logp_pi``  (batch,)          | Gives log probability, according to
                                               | the policy, of the action sampled by
                                               | ``pi``.
                ``v``        (batch,)          | Gives the value estimate for states
                                               | in ``x_ph``. (Critical: make sure 
                                               | to flatten this!)
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
                function you provided to PPO.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.

            gamma (float): Discount factor. (Always between 0 and 1.)

            clip_ratio (float): Hyperparameter for clipping in the policy objective.
                Roughly: how far can the new policy go from the old policy while 
                still profiting (improving the objective function)? The new policy 
                can still go farther than the clip_ratio says, but it doesn't help
                on the objective anymore. (Usually small, 0.1 to 0.3.)

            pi_lr (float): Learning rate for policy optimizer.

            vf_lr (float): Learning rate for value function optimizer.

            train_pi_iters (int): Maximum number of gradient descent steps to take 
                on policy loss per epoch. (Early stopping may cause optimizer
                to take fewer than this.)

            train_v_iters (int): Number of gradient descent steps to take on 
                value function per epoch.

            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update. This will get used 
                for early stopping. (Usually small, 0.01 or 0.05.)

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

        self.target_kl = target_kl
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters 
        self.obs_dim = obs_dim 
        self.act_dim = act_dim
        self.buff_size = buff_size
        self.clip_ratio = clip_ratio
        self.delta=delta
        self.clip_ratio=clip_ratio
        self.damping_coeff=damping_coeff
        self.cg_iters=cg_iters
        self.backtrack_iters=backtrack_iters
        self.backtrack_coeff=backtrack_coeff
        self.algo = algo
        # Inputs to computation graph
        #####################################change here
        self.x_ph, self.a_ph = core.placeholders(self.obs_dim, self.act_dim)
        #####################################change here
        self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)

        # Main outputs from computation graph
        self.pi, self.logp, self.logp_pi, self.info, self.info_phs, self.d_kl, self.v = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph] + core.values_as_sorted_list(self.info_phs)

        # Every step, get: action, value, and logprob
        self.get_action_ops = [self.pi, self.v, self.logp_pi] + core.values_as_sorted_list(self.info)

        # Experience buffer
        info_shapes = {k: v.shape.as_list()[1:] for k,v in self.info_phs.items()}
        self.buf = TRPOBuffer(self.obs_dim, self.act_dim, buff_size, info_shapes, gamma, lam)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # TRPO objectives
        ratio = tf.exp(self.logp - self.logp_old_ph)          # pi(a|s) / pi_old(a|s)
        self.pi_loss = -tf.reduce_mean(ratio * self.adv_ph)
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        
        # Optimizer for value function
        self.train_v = tf.train.AdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_loss)

        # Symbols needed for CG solver
        pi_params = core.get_vars('pi')
        self.gradient = core.flat_grad(self.pi_loss, pi_params)
        self.v_ph, self.hvp = core.hessian_vector_product(self.d_kl, pi_params)
        if damping_coeff > 0:
            self.hvp += damping_coeff * self.v_ph

        # Symbols for getting and setting params
        self.get_pi_params = core.flat_concat(pi_params)
        self.set_pi_params = core.assign_params_from_flat(self.v_ph, pi_params)

        self.start_time = time.time()
        #sess = tf.Session()
        #sess.run(tf.global_variables_initializer())

        # Sync params across processes
        #sess.run(sync_all_params())

        # Setup model saving
        #logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def cg(self, Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        print(x.shape)
        r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r,r)
        for _ in range(self.cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r,r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

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
        self.logger.log_tabular('KL', average_only=True)
        if self.algo=='trpo':
            self.logger.log_tabular('BacktrackIters', average_only=True)
        self.logger.log_tabular('Time', time.time()-self.start_time)
        self.logger.dump_tabular()

    def update(self):
        # Prepare hessian func, gradient eval
        inputs = {k:v for k,v in zip(self.all_phs, self.buf.get())}
        #Hx = lambda x : mpi_avg(self.sess.run(self.hvp, feed_dict={**inputs, v_ph: x}))
        Hx = lambda x : mpi_avg(self.sess.run(self.hvp, feed_dict={**inputs, self.v_ph: x}))
        g, pi_l_old, v_l_old = self.sess.run([self.gradient, self.pi_loss, self.v_loss], feed_dict=inputs)
        #g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)
        g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)
        print(g.shape)

        # Core calculations for TRPO or NPG
        x = self.cg(Hx, g)
        alpha = np.sqrt(2*self.delta/(np.dot(x, Hx(x))+EPS))
        old_params = self.sess.run(self.get_pi_params)

        def set_and_eval(step):
            self.sess.run(self.set_pi_params, feed_dict={self.v_ph: old_params - alpha * x * step})
            #return mpi_avg(sess.run([d_kl, pi_loss], feed_dict=inputs))
            return mpi_avg(self.sess.run([self.d_kl, self.pi_loss], feed_dict=inputs))
            
        if self.algo=='npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = set_and_eval(step=1.)

        elif self.algo=='trpo':
            # trpo augments npg with backtracking line search, hard kl
            for j in range(self.backtrack_iters):
                kl, pi_l_new = set_and_eval(step=self.backtrack_coeff**j)
                if kl <= self.delta and pi_l_new <= pi_l_old:
                    self.logger.log('Accepting new params at step %d of line search.'%j)
                    self.logger.store(BacktrackIters=j)
                    break

                if j==self.backtrack_iters-1:
                    self.logger.log('Line search failed! Keeping old params.')
                    self.logger.store(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(step=0.)

        # Value function updates
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)
        v_l_new = self.sess.run(self.v_loss, feed_dict=inputs)

        # Log changes from update
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))


    def get_logger(self):
        return self.logger