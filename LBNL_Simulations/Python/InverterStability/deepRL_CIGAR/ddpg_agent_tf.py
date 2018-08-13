import tensorflow as tf
import numpy as np
import copy

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size


        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: x if x is None else tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 700)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        
        net = tflearn.fully_connected(net, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        
        net = tflearn.fully_connected(net, 200)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='sigmoid', weights_init=w_init)
        scaled_out = tf.multiply(out, self.action_bound)

        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None,])
        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 700)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        
        net = tflearn.fully_connected(net, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 200)
        t2 = tflearn.fully_connected(action, 200)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class Agent(object):

    """Summary
    
    Attributes:
        a (TYPE): Description
        a_dim (np.array): action dimension
        action_bound (int): action bound (output will be from 0 -> action_bound)
        actions (list): actions of agent from all timestep in an episode
        actor (actorNetwork): actor network
        actor_lr (float): actor learning rate
        actor_noise (noise object): noise adding to action
        agent (int): agent code
        allReward (list): all the rewards from different episodes (simulations)
        attack (bool): attack agent or defense agent - the difference is the reward function
        batch_size (int): size of a batch
        buffer_size (int): size of the buffer
        critic (criticNetwork): critic network
        critic_lr (float): critic learning rate
        ep_reward (float): accumulated reward in that episode
        epsilon (float): decreasing exporation overtime
        EXPLORE (TYPE): stop exploring after EXPLORE accumulated timestep
        gamma (TYPE): Description
        random_seed (TYPE): Description
        replay_buffer (replay_buffer): buffer contains experiences
        reward (float): reward for the agent
        s_dim (int): dimension of a state
        sess (sess): running tensorflow session
        tau (float): Description
        voltage (list): all voltage of this agent for the episode
    """
    
    def __init__(self, sess, agent_code, attack=False, state_dim=9, action_dim=4, action_bound=0.15,
                 actor_lr=0.0001, critic_lr=0.001, tau=0.001, gamma=0.99, batch_size=50, buffer_size=5000, random_seed=1234, EXPLORE=144100):
        self.agent = agent_code
        self.sess = sess
        self.attack = attack
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.random_seed = random_seed
        self.EXPLORE = EXPLORE
        self.epsilon = 1
        self.ep_reward = 0
        self.actions =[]
        self.voltage= []
        self.allReward=[]
        self.reward = 0
        
        #create actor
        self.actor = ActorNetwork(self.sess, self.s_dim, self.a_dim, self.action_bound,
                                  self.actor_lr, self.tau, self.batch_size)
        
        #create critic
        self.critic = CriticNetwork(self.sess, self.s_dim, self.a_dim,
                               self.critic_lr, self.tau,
                               self.gamma,
                               self.actor.get_num_trainable_vars())

        self.sess.run(tf.global_variables_initializer())
        self.actor.update_target_network()
        self.critic.update_target_network()
        
        #create actor noise
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim), sigma=0.02, theta=.1, dt=1e-2)
        
        #create buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.random_seed)
        
        
    def observe_reward(self, yk):
        """Summary
        
        Args:
            yk (float): output of the voltage observer
        
        Returns:
            TYPE: reward for the agent (depends on attacker or defender, reward would be y_k or -y_k)
        """
        if self.attack:
            self.reward = yk[self.agent]
        else:
            if yk[self.agent] >0.25:
                self.reward = -yk[self.agent]
            else:
                self.reward = 0
        return self.reward
    
    def get_action(self, state):
        """Summary
        get the action by the actor Network under a state
        
        Returns:
            np.array: array of 4 points (actions of the agent)
        """
        #decay exploration
        s = state.get_state_agent(self.agent)
        self.epsilon -= 1/self.EXPLORE
        self.a = self.actor.predict(np.reshape(s, (1, self.actor.s_dim))) + self.actor_noise() * max(self.epsilon, 0)
        return self.a[0]
    
    def add_experience_to_buffer(self, s, a, r, t, s2):
        """Summary
        adding experience to the buffer
        """
        self.replay_buffer.add(np.reshape(s.get_state_agent(self.agent), (self.actor.s_dim,)), np.reshape(a, (self.actor.a_dim,)), self.observe_reward(r),
                              t, np.reshape(s2.get_state_agent(self.agent), (self.actor.s_dim,)))
        
    def train(self):
        """Summary
        1. train the agent with a batch of experience getting from the buffer
        2. update the actor and critic network
        """
        if self.replay_buffer.size() > self.batch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.batch_size)
            target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))
        
            y_i = []
            for k in range(self.batch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.critic.gamma * target_q[k])
            
            predicted_q_value, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i, self.batch_size, 1))

        
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()
    
    def summaries(self, terminal, voltage, action):
        """Summary
        summaries of Voltage, Actions taken, Reward of the agent
        """
        if terminal == False:
            self.ep_reward += self.reward
            self.actions.append(copy.deepcopy(action))
            self.voltage.append(voltage[self.agent])
        else:
            self.ep_reward += self.reward
            self.allReward.append(self.ep_reward)
            self.actions.append(action)
            self.voltage.append(voltage[self.agent])
            
            #get the value
            allR = copy.deepcopy(self.allReward)
            allA = copy.deepcopy(self.actions)
            allV = copy.deepcopy(self.voltage)
            #reset for new episode
            self.actions = []
            self.ep_reward = 0
            self.voltage = []
            return allV, allA, allR