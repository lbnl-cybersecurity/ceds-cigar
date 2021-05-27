import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
from pycigar.utils.input_parser import input_parser
import numpy as np
from pycigar.utils.registry import register_devcon
import tensorflow as tf
from ray.rllib.models.catalog import ModelCatalog
from gym.spaces import Tuple, Discrete, Box
import matplotlib
from pycigar.utils.output import plot_new
from pycigar.utils.registry import make_create_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import numpy as np
import gym
from gym import wrappers
import pandas as pd
from pycigar.envs import CentralControlPhaseSpecificContinuousPVInverterEnv


misc_inputs = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
dss = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
load_solar = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
breakpoints = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'


start = 100
hack = 0.3
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=hack)
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
del sim_params['attack_randomization']
for node in sim_params['scenario_config']['nodes']:
    node['devices'][0]['adversary_controller'] =  'adaptive_fixed_controller'
sim_params['M'] = 300  # oscillation penalty
sim_params['N'] = 0.5  # initial action penalty
sim_params['P'] = 1    # last action penalty
sim_params['Q'] = 1    # power curtailment penalty
sim_params['T'] = 150  # imbalance penalty
# pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
#                   'env_name': 'CentralControlPhaseSpecificContinuousPVInverterEnv',
#                   'simulator': 'opendss'}

# create_env, env_name = make_create_env(pycigar_params, version=0)
# register_env(env_name, create_env)


#class for normalizing the observations
class Normalizer():
    #Welford's online algorithm
    #implementation from: iamsuvhro
    def __init__(self, n_inputs):
        self.mean = np.zeros(n_inputs)
        self.n = np.zeros(n_inputs)
        self.sos_diff = np.zeros(n_inputs)
        self.var = np.zeros(n_inputs)

    def update_statistics(self, x):
        self.n += 1
        #update mean 
        last_mean = self.mean.copy()
        self.mean += (x - self.mean)/self.n
        #update sum of squares differences
        self.sos_diff += (x-last_mean)*(x-self.mean)
        self.var = (self.sos_diff/self.n).clip(min=1e-2)

    def normalize(self, u):
        self.update_statistics(u)
        u_no_mean = u - self.mean
        u_std = np.sqrt(self.var)
        return u_no_mean/u_std

#class for ARS agent
class ARSAgent():
    def __init__(self):
        self.alpha = 0.002 #learning rate
        self.mu = 0.03 #exploration noise
        self.num_directions = 4 #number of random directions to consider
        self.num_best_directions = 4 #number of best directions to consider
        assert self.num_best_directions <= self.num_directions
        self.max_iterations = 20 #number of iterations
        self.env = CentralControlPhaseSpecificContinuousPVInverterEnv(sim_params=sim_params)
        self.n_inputs = self.env.observation_space.shape[0]
        self.n_outputs = 3
        self.seed = 1
        np.random.seed(self.seed)
        self.theta = np.zeros((self.n_inputs,self.n_outputs))
        self.normalizer = Normalizer(self.n_inputs)

    def get_action(self, state, theta):
        _u = np.dot(theta.T, state)
        return _u
        # if _u > 10:
        #     return 1
        # else:
        #     return 0

    def rollout(self, theta):
        state = self.env.reset()
        #rollout for episode:
        done = False
        sum_rewards = 0
        k = 0
        while not done:
            #normalize state
            state = self.normalizer.normalize(state)
            #get next action
            u = self.get_action(state, theta)
            state, reward, done, _ = self.env.step(u)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            k+=1
        return sum_rewards

    def random_directions(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.num_directions)]

    def random_search(self):
        #run 1 iteration of augmented random search
        d = self.random_directions()
        r_pos = []
        r_neg = []
        for i in range(0,self.num_directions):
            #generate random direction
            _d = d[i]
            #rollout in _d and -_d
            theta_d_pos = self.theta + self.mu * _d
            theta_d_neg = self.theta - self.mu * _d
            #compute positive and negative rewards
            r_pos.append(self.rollout(theta_d_pos))
            r_neg.append(self.rollout(theta_d_neg))

        #compute std for rewards
        r_std = np.asarray(r_pos + r_neg).std()

        #find indices of best b rewards
        best_scores = [max(_r_pos, _r_neg) for k,(_r_pos,_r_neg) in enumerate(zip(r_pos, r_neg))]
        idxs = np.asarray(best_scores).argsort()[-self.num_best_directions:]
        #GD
        _theta = np.zeros(self.theta.shape)
        for idx in list(idxs):
            _theta += self.alpha/self.num_best_directions * (r_pos[idx] - r_neg[idx])/r_std * d[idx]
        #update theta
        self.theta += _theta
        #rollout with the new policy for evaluation
        r_eval = self.rollout(self.theta)
        return _theta, r_eval

    def train(self):
        k=0
        thetas = []
        rewards = []
        while k < self.max_iterations:
            #run step of ARS
            _theta, _r = self.random_search()
            thetas.append(_theta)
            rewards.append(_r)
            print("Iteration: ", k, " ---------- reward: ", _r)
            k+=1
        return thetas, rewards

#create ARS agent
ars = ARSAgent()
#train the agent
thetas,rewards = ars.train()

#store results in dataframe and save to csv
iteration = [i for i in range(1,len(rewards)+1)]
rewards_dict = {'rewards':rewards}
rewards_df = pd.DataFrame(rewards_dict)
rewards_df.to_csv('ARS_cartpole_rewards.csv')

#save best weights to csv
np.savetxt('ARS_cartpole_theta.csv', thetas[-1], delimiter=',')
#to load use: np.loadtxt('ARS_theta.csv',delimiter=',')