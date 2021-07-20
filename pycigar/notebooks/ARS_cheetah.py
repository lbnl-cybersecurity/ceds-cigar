import matplotlib.pyplot as plt
#plt.switch_backend('tkagg')
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
import argparse
import os
from pycigar.utils.output import plot_new
from pycigar.utils.logging import logger
import pycigar
import pybullet_envs

def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run distributed runs to better understand PyCIGAR hyperparameters')
    parser.add_argument('--save-path', type=str, default='~/ars_cheetah_org', help='where to save the results')

    return parser.parse_args()


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
        self.alpha = 0.02 #learning rate
        self.mu = 0.03 #exploration noise
        self.num_directions = 16 #number of random directions to consider
        self.num_best_directions = 16 #number of best directions to consider
        assert self.num_best_directions <= self.num_directions
        self.max_iterations = 1000 #number of iterations
        self.max_episode_steps = 1000
        self.env = gym.make('HalfCheetahBulletEnv-v0')
        self.n_inputs = self.env.observation_space.shape[0]
        self.n_layer1 = 5
        self.n_outputs = self.env.action_space.shape[0]
        self.seed = 1
        np.random.seed(self.seed)
        self.theta1 = np.zeros((self.n_inputs, self.n_outputs))
        self.normalizer = Normalizer(self.n_inputs)

    def get_action(self, state, theta1):
        _u = np.dot(theta1.T, state)
        return _u

    def rollout(self, theta1):
        state = self.env.reset()
        #rollout for episode:
        done = False
        sum_rewards = 0
        while not done:
            #normalize state
            state = self.normalizer.normalize(state)
            #get next action
            u = self.get_action(state, theta1)
            state, reward, done, _ = self.env.step(u)
            #reward = max(min(reward, 1), -1)
            sum_rewards += reward
        return sum_rewards

    def random_directions(self):
        return [np.random.randn(*self.theta1.shape) for _ in range(self.num_directions)]

    def random_search(self):
        #run 1 iteration of augmented random search
        d1 = self.random_directions()
        r_pos = []
        r_neg = []
        for i in range(0,self.num_directions):
            print(" i ", i)
            #generate random direction
            _d1 = d1[i]
            #rollout in _d and -_d
            theta_d1_pos = self.theta1 + self.mu * _d1
            theta_d1_neg = self.theta1 - self.mu * _d1
            #compute positive and negative rewards
            r_pos.append(self.rollout(theta_d1_pos))
            r_neg.append(self.rollout(theta_d1_neg))

        #compute std for rewards
        r_std = np.asarray(r_pos + r_neg).std()
        r_std = max(r_std, 0.0001) # at optimal strategy, r_std can be zero, thus throwing deviding by zero error
        #find indices of best b rewards
        best_scores = [max(_r_pos, _r_neg) for k,(_r_pos,_r_neg) in enumerate(zip(r_pos, r_neg))]
        idxs = np.asarray(best_scores).argsort()[-self.num_best_directions:]
        #GD
        _theta1 = np.zeros(self.theta1.shape)
        for idx in list(idxs):
            _theta1 += self.alpha/self.num_best_directions * (r_pos[idx] - r_neg[idx])/r_std * d1[idx]
        #update theta
        self.theta1 += _theta1
        #rollout with the new policy for evaluation
        r_eval = self.rollout(self.theta1)
        return self.theta1, r_eval

    def evaluation(self, theta1, iteration):
        self.rollout(theta1)
        Logger = logger()
        save_path = os.path.expanduser(args.save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f = plot_new(Logger.log_dict, Logger.custom_metrics, iteration, False)
        f.savefig(os.path.join(save_path, 'eval-epoch-' + str(iteration) + '.png'), bbox_inches='tight')
        plt.close(f)

    def train(self):
        k=0
        thetas1 = []
        rewards = []
        while k < self.max_iterations:
            #run step of ARS
            print("Start ", k, end = " ")
            _theta1, _r = self.random_search()
            thetas1.append(_theta1)
            rewards.append(_r)
            print("Iteration: ", k, " ---------- reward: ", _r)
            k+=1
            #self.evaluation(_theta1, _theta2, k)

        return thetas1, rewards

if __name__ == '__main__':


    args = parse_cli_args()
    #create ARS agent
    ars = ARSAgent()
    #train the agent
    thetas1, rewards = ars.train()

    #store results in dataframe and save to csv
    iteration = [i for i in range(1,len(rewards)+1)]
    rewards_dict = {'rewards':rewards}
    rewards_df = pd.DataFrame(rewards_dict)

    save_path = os.path.expanduser(args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rewards_df.to_csv(os.path.join(save_path, 'ARS_rewards.csv'))
    #save best weights to csv
    np.savetxt(os.path.join(save_path, 'ARS_cheetah_theta1.csv'), thetas1[-1], delimiter=',')
    fig1 = plt.figure(figsize = [16, 4])
    plt.plot(rewards,label="rewards")
    plt.grid()
    plt.title('Reward')
    plt.xlabel('Time [s]')
    plt.ylabel('Reward')
    plt.legend()
    fig1.savefig(os.path.join(save_path, 'reward.png'))