
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 09:59:30 2018

@author: Sy-Toan Ngo
"""

import numpy as np
import tensorflow as tf


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    #x = tf.layers.Flatten()(x)
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.nn.relu, 
                     output_activation=tf.tanh, act_limit=0.2):
    
    act_dim = a.shape.as_list()[-1]
    
    x = tf.layers.Flatten()(x)
    
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, q, q_pi