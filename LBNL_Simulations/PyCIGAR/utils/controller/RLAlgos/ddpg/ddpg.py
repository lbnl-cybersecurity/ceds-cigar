import numpy as np
import tensorflow as tf
import gym
import time
import utils.controller.RLAlgos.ddpg.core as core
from utils.controller.RLAlgos.ddpg.core import get_vars
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def current_size(self):
        return self.size

    def get(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return [self.obs1_buf[idxs], self.obs2_buf[idxs], self.acts_buf[idxs], 
                self.rews_buf[idxs], self.done_buf[idxs]]

"""

Deep Deterministic Policy Gradient (DDPG)

"""

class DDPG:
    def __init__(self, sess, logger, obs_dim=(60,3), act_dim=(4,), act_limit=0.2, 
             actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
             replay_size=int(1e6), gamma=0.99, 
             polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=2, start_steps=2, 
             act_noise=0.1, act_shift=1, save_freq=1):
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
                ``pi``       (batch, act_dim)  | Deterministically computes actions
                                               | from policy given states.
                ``q``        (batch,)          | Gives the current estimate of Q* for 
                                               | states in ``x_ph`` and actions in
                                               | ``a_ph``.
                ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                                               | ``pi`` for states in ``x_ph``: 
                                               | q(x, pi(x)).
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
                function you provided to DDPG.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            pi_lr (float): Learning rate for policy.

            q_lr (float): Learning rate for Q-networks.

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            act_noise (float): Stddev for Gaussian exploration noise added to 
                policy at training time. (At test time, no noise is added.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        self.sess = sess
        self.logger = logger

        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.obs_dim = obs_dim 
        self.act_dim = act_dim
        self.replay_size = replay_size
        self.act_noise = act_noise
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = act_limit
        self.warmUp = 0
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.act_shift = act_shift
        # Share information about action space with policy architecture

        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
        self.all_phs = [self.x_ph, self.x2_ph, self.a_ph, self.r_ph, self.d_ph]

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.pi, self.q, self.q_pi = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)
        
        # Target networks
        with tf.variable_scope('target'):
            # Note that the action placeholder going to actor_critic here is 
            # irrelevant, because we only need q_targ(s, pi_targ(s)).
            self.pi_targ, _, self.q_pi_targ  = actor_critic(self.x2_ph, self.a_ph, **ac_kwargs, act_limit=self.act_limit)

        # Experience buffer
        self.buf = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
        print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

        # Bellman backup for Q function
        backup = tf.stop_gradient(self.r_ph + gamma*(1-self.d_ph)*self.q_pi_targ)

        # DDPG losses
        self.pi_loss = -tf.reduce_mean(self.q_pi)
        self.q_loss = tf.reduce_mean((self.q-backup)**2)

        # Separate train ops for pi, q
        self.train_pi_op = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss, var_list=get_vars('main/pi'))
        self.train_q_op = tf.train.AdamOptimizer(learning_rate=q_lr).minimize(self.q_loss, var_list=get_vars('main/q'))

        # Polyak averaging for target variables
        self.target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.start_time = time.time()
        #sess = tf.Session()
        #sess.run(tf.global_variables_initializer())
        #sess.run(target_init)

        # Setup model saving
        #logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q': q})

    def update(self):
        inputs = {k:v for k,v in zip(self.all_phs, self.buf.get(self.batch_size))}
        #print(inputs)
        #pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)
        # Q-learning update
        outs = self.sess.run([self.q_loss, self.q, self.train_q_op], feed_dict=inputs)
        self.logger.store(LossQ=outs[0], QVals=outs[1])
        
        # Policy update
        outs = self.sess.run([self.pi_loss, self.train_pi_op, self.target_update], feed_dict=inputs)
        self.logger.store(LossPi=outs[0])


    def dump_logger(self):
        #self.logger.log_tabular('Epoch', epoch)
        #self.logger.log_tabular('EpRet', with_min_and_max=True)
        #self.logger.log_tabular('EpLen', average_only=True)
        #self.logger.log_tabular('VVals', with_min_and_max=True)
        #logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Time', time.time()-self.start_time)
        self.logger.dump_tabular()