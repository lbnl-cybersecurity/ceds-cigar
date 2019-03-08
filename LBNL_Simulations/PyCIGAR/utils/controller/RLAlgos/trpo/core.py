
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 09:59:30 2018

@author: Sy-Toan Ngo
"""


import numpy as np
import tensorflow as tf
import scipy.signal

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))

def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    x = tf.layers.Flatten()(x)
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    """
    tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    """
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1- mu0)**2 + var0)/(var1 + EPS) - 1) +  log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)

def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

def flat_grad(f, params):
    return flat_concat(tf.gradients(xs=params, ys=f))

def hessian_vector_product(f, params):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    x = tf.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g*x), params)

def assign_params_from_flat(x, params):
    flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    mu = tf.multiply(mu, 0.2) + 1
    log_std = tf.get_variable(name='log_std', initializer=-3.*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    
    old_mu_ph, old_log_std_ph = placeholders(act_dim, act_dim)
    d_kl = diagonal_gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)

    info = {'mu': mu, 'log_std': log_std}
    info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

    return pi, logp, logp_pi, info, info_phs, d_kl



"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=tf.tanh, policy=None):

    # default policy builder depends on action space
    if policy is None:
        policy = mlp_gaussian_policy

    with tf.variable_scope('pi'):
        policy_outs = policy(x, a, hidden_sizes, activation, output_activation)
        pi, logp, logp_pi, info, info_phs, d_kl = policy_outs
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    
    return pi, logp, logp_pi, info, info_phs, d_kl, v