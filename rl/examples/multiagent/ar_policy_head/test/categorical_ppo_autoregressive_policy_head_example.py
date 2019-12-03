"""Experiment of categorical autoregressive policy head with PPO.
We create a simple environment as following:
    - Randomly generate state [0, 1] every timestep.
    - Each action is in [0, 1, 2, 3, 4, 5]
    - The action is a list of [a1, a2, a3, a4, a5].
    - If a1 == laststate -> reward + 1
    - If a1 < a2 < a3 < a4, we get the best reward is 10.

In the policy loss, we add a categorical loss when there is a violation in action.
a1 > a2, a2 > a3, a3 > a4, a4 > a5.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.spaces import Discrete, Tuple
import argparse
import random

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import ActionDistribution
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.policy import TupleActions
from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.utils.annotations import override, DeveloperAPI

DISCRETIZE = 10

tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=210)


class Categorical(TFActionDistribution):
    """Categorical distribution for discrete action spaces."""

    @DeveloperAPI
    def __init__(self, inputs, model=None):
        super(Categorical, self).__init__(inputs, model)

    @override(ActionDistribution)
    def logp(self, x):
        # print(self.inputs)
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.inputs, labels=tf.cast(x, tf.int32))

    @override(ActionDistribution)
    def entropy(self):
        a0 = self.inputs - tf.reduce_max(
            self.inputs, reduction_indices=[1], keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, reduction_indices=[1], keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), reduction_indices=[1])

    @override(ActionDistribution)
    def kl(self, other):
        a0 = self.inputs - tf.reduce_max(
            self.inputs, reduction_indices=[1], keep_dims=True)
        a1 = other.inputs - tf.reduce_max(
            other.inputs, reduction_indices=[1], keep_dims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, reduction_indices=[1], keep_dims=True)
        z1 = tf.reduce_sum(ea1, reduction_indices=[1], keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(
            p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), reduction_indices=[1])

    @override(TFActionDistribution)
    def _build_sample_op(self):
        return tf.squeeze(tf.multinomial(self.inputs, 1), axis=1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        return action_space.n


class AutoregressiveOutput(ActionDistribution):
    """Action distribution P(a1, a2, a3, a4, a5) = P(a1) * P(a2 | a1) * P(a3 | a1, a2)..."""
    @staticmethod
    def required_model_output_shape(self, model_config):
        return 16  # controls model output feature vector size

    def sample(self):
        # first, sample a1
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # sample a2 conditioned on a1
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()

        # sample a2 conditioned on a1
        a3_dist = self._a3_distribution(a1, a2)
        a3 = a3_dist.sample()

        # sample a2 conditioned on a1
        a4_dist = self._a4_distribution(a1, a2, a3)
        a4 = a4_dist.sample()

        # sample a2 conditioned on a1
        a5_dist = self._a5_distribution(a1, a2, a3, a4)
        a5 = a5_dist.sample()

        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2) + a3_dist.logp(a3) + a4_dist.logp(a4) + a5_dist.logp(a5)

        # return the action tuple
        return TupleActions([a1, a2, a3, a4, a5])

    def logp(self, actions):
        a1, a2, a3, a4, a5 = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3], actions[:, 4]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        a3_vec = tf.expand_dims(tf.cast(a3, tf.float32), 1)
        a4_vec = tf.expand_dims(tf.cast(a4, tf.float32), 1)
        a1_logits, a2_logits, a3_logits, a4_logits, a5_logits = self.model.action_model([self.inputs, a1_vec,  a2_vec,  a3_vec,  a4_vec])

        return (Categorical(a1_logits).logp(a1) + Categorical(a2_logits).logp(a2) +
                Categorical(a3_logits).logp(a3) + Categorical(a4_logits).logp(a4) +
                Categorical(a5_logits).logp(a5))

    def sampled_action_logp(self):
        return tf.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        a3_dist = self._a3_distribution(a1_dist.sample(), a2_dist.sample())
        a4_dist = self._a4_distribution(a1_dist.sample(), a2_dist.sample(), a3_dist.sample())
        a5_dist = self._a5_distribution(a1_dist.sample(), a2_dist.sample(), a3_dist.sample(), a4_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy() + a3_dist.entropy() + a4_dist.entropy() + a5_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_dist = self._a2_distribution(a1)
        a2_terms = a2_dist.kl(other._a2_distribution(a1))

        a2 = a2_dist.sample()
        a3_dist = self._a3_distribution(a1, a2)
        a3_terms = a3_dist.kl(other._a3_distribution(a1, a2))

        a3 = a3_dist.sample()
        a4_dist = self._a4_distribution(a1, a2, a3)
        a4_terms = a4_dist.kl(other._a4_distribution(a1, a2, a3))

        a4 = a4_dist.sample()
        a5_dist = self._a5_distribution(a1, a2, a3, a4)
        a5_terms = a5_dist.kl(other._a5_distribution(a1, a2, a3, a4))

        return a1_terms + a2_terms + a3_terms + a4_terms + a5_terms

    def _a1_distribution(self):
        BATCH = tf.shape(self.inputs)[0]
        a1_logits, _, _, _, _ = self.model.action_model([self.inputs, tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1))])
        a1_dist = Categorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        BATCH = tf.shape(self.inputs)[0]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        _, a2_logits, _, _, _ = self.model.action_model([self.inputs, a1_vec, tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1))])

        # create mask [0, 1, 2, 3, 4, 5] size 6 of the output action
        zeros = tf.cast(tf.zeros_like(a2_logits), tf.float32)  # [BATCH, 6]
        mask_tensor = tf.add(zeros, tf.cast(tf.constant(list(range(DISCRETIZE))), tf.float32))
        mask_a = tf.transpose(tf.add(tf.transpose(zeros), tf.cast(a1, tf.float32)))
        mask_bool = tf.greater_equal(mask_tensor, mask_a)
        inf = tf.add(zeros, tf.constant([-1e10]*DISCRETIZE))
        a2_logits = tf.where(mask_bool, a2_logits, inf)
        a2_dist = Categorical(a2_logits)
        return a2_dist

    def _a3_distribution(self, a1, a2):
        BATCH = tf.shape(self.inputs)[0]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        _, _, a3_logits, _, _ = self.model.action_model([self.inputs, a1_vec, a2_vec, tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1))])

        # create mask [0, 1, 2, 3, 4, 5] size 6 of the output action
        zeros = tf.cast(tf.zeros_like(a3_logits), tf.float32)  # [BATCH, 6]
        mask_tensor = tf.add(zeros, tf.cast(tf.constant(list(range(DISCRETIZE))), tf.float32))
        mask_a = tf.transpose(tf.add(tf.transpose(zeros), tf.cast(a2, tf.float32)))
        mask_bool = tf.greater_equal(mask_tensor, mask_a)
        inf = tf.add(zeros, tf.constant([-1e10]*DISCRETIZE))
        a3_logits = tf.where(mask_bool, a3_logits, inf)
        a3_dist = Categorical(a3_logits)
        return a3_dist

    def _a4_distribution(self, a1, a2, a3):
        BATCH = tf.shape(self.inputs)[0]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        a3_vec = tf.expand_dims(tf.cast(a3, tf.float32), 1)
        _, _, _, a4_logits, _ = self.model.action_model([self.inputs, a1_vec, a2_vec, a3_vec, tf.zeros((BATCH, 1))])

        # create mask [0, 1, 2, 3, 4, 5] size 6 of the output action
        zeros = tf.cast(tf.zeros_like(a4_logits), tf.float32)  # [BATCH, 6]
        mask_tensor = tf.add(zeros, tf.cast(tf.constant(list(range(DISCRETIZE))), tf.float32))
        mask_a = tf.transpose(tf.add(tf.transpose(zeros), tf.cast(a3, tf.float32)))
        mask_bool = tf.greater_equal(mask_tensor, mask_a)
        inf = tf.add(zeros, tf.constant([-1e10]*DISCRETIZE))
        a4_logits = tf.where(mask_bool, a4_logits, inf)
        a4_dist = Categorical(a4_logits)
        return a4_dist

    def _a5_distribution(self, a1, a2, a3, a4):
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        a3_vec = tf.expand_dims(tf.cast(a3, tf.float32), 1)
        a4_vec = tf.expand_dims(tf.cast(a4, tf.float32), 1)
        _, _, _, _, a5_logits = self.model.action_model([self.inputs, a1_vec, a2_vec, a3_vec, a4_vec])

        # create mask [0, 1, 2, 3, 4, 5] size 6 of the output action
        zeros = tf.cast(tf.zeros_like(a5_logits), tf.float32)
        mask_tensor = tf.add(zeros, tf.cast(tf.constant(list(range(DISCRETIZE))), tf.float32))
        mask_a = tf.transpose(tf.add(tf.transpose(zeros), tf.cast(a4, tf.float32)))
        mask_bool = tf.greater_equal(mask_tensor, mask_a)
        inf = tf.add(zeros, tf.constant([-1e10]*DISCRETIZE))
        a5_logits = tf.where(mask_bool, a5_logits, inf)
        a5_dist = Categorical(a5_logits)
        return a5_dist


class AutoregressiveActionsModel(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(AutoregressiveActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        if action_space != Tuple([Discrete(DISCRETIZE), Discrete(DISCRETIZE), Discrete(DISCRETIZE), Discrete(DISCRETIZE), Discrete(DISCRETIZE)]):
            raise ValueError("This model only supports the [b, b, b, b, b] action space")

        # Inputs
        obs_input = tf.keras.layers.Input(shape=obs_space.shape, name="obs_input")
        a1_input = tf.keras.layers.Input(shape=(1, ), name="a1_input")
        a2_input = tf.keras.layers.Input(shape=(1, ), name="a2_input")
        a3_input = tf.keras.layers.Input(shape=(1, ), name="a3_input")
        a4_input = tf.keras.layers.Input(shape=(1, ), name="a4_input")

        ctx_input = tf.keras.layers.Input(shape=(num_outputs, ), name="ctx_input")

        # Output of the model (normally 'logits', but for an autoregressive
        # dist this is more like a context/feature layer encoding the obs)
        context = tf.keras.layers.Dense(
            num_outputs,
            name="hidden",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(obs_input)

        # V(s)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(context)

        # P(a1 | obs)
        a1_logits = tf.keras.layers.Dense(DISCRETIZE, name="a1_logits", activation=None, kernel_initializer=normc_initializer(0.01))(ctx_input)

        # P(a2 | a1, obs)
        a2_context = tf.keras.layers.Concatenate(axis=1)([ctx_input, a1_input])
        a2_hidden = tf.keras.layers.Dense(32, name="a2_hidden", activation=tf.nn.tanh, kernel_initializer=normc_initializer(1.0))(a2_context)
        a2_logits = tf.keras.layers.Dense(DISCRETIZE, name="a2_logits", activation=None, kernel_initializer=normc_initializer(0.01))(a2_hidden)

        # P(a3 | a1, a2, obs)
        a3_context = tf.keras.layers.Concatenate(axis=1)([ctx_input, a1_input, a2_input])
        a3_hidden = tf.keras.layers.Dense(32, name="a3_hidden", activation=tf.nn.tanh, kernel_initializer=normc_initializer(1.0))(a3_context)
        a3_logits = tf.keras.layers.Dense(DISCRETIZE, name="a3_logits", activation=None, kernel_initializer=normc_initializer(0.01))(a3_hidden)

        # P(a4 | a1, a2, a3, obs)
        a4_context = tf.keras.layers.Concatenate(axis=1)([ctx_input, a1_input, a2_input, a3_input])
        a4_hidden = tf.keras.layers.Dense(32, name="a4_hidden", activation=tf.nn.tanh, kernel_initializer=normc_initializer(1.0))(a4_context)
        a4_logits = tf.keras.layers.Dense(DISCRETIZE, name="a4_logits", activation=None, kernel_initializer=normc_initializer(0.01))(a4_hidden)

        # P(a5 | a1, a2, a3, a4, obs)
        a5_context = tf.keras.layers.Concatenate(axis=1)([ctx_input, a1_input, a2_input, a3_input, a4_input])
        a5_hidden = tf.keras.layers.Dense(32, name="a5_hidden", activation=tf.nn.tanh, kernel_initializer=normc_initializer(1.0))(a5_context)
        a5_logits = tf.keras.layers.Dense(DISCRETIZE, name="a5_logits", activation=None, kernel_initializer=normc_initializer(0.01))(a5_hidden)

        # Base layers
        self.base_model = tf.keras.Model(obs_input, [context, value_out])
        self.register_variables(self.base_model.variables)
        # self.base_model.summary()

        # Autoregressive action sampler
        self.action_model = tf.keras.Model(inputs=[ctx_input, a1_input, a2_input, a3_input, a4_input],
                                           outputs=[a1_logits, a2_logits, a3_logits, a4_logits, a5_logits])
        # self.action_model.summary()
        self.register_variables(self.action_model.variables)

    def forward(self, input_dict, state, seq_lens):
        context, self._value_out = self.base_model(input_dict["obs"])
        return context, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class CorrelatedCategoricalActionsEnv(gym.Env):
    """Simple env in which the policy has to emit a tuple of equal actions.
    The best score would be ~210 reward."""

    def __init__(self, _):
        self.observation_space = Discrete(2)
        self.action_space = Tuple([Discrete(DISCRETIZE), Discrete(DISCRETIZE), Discrete(DISCRETIZE), Discrete(DISCRETIZE), Discrete(DISCRETIZE)])

    def reset(self):
        self.t = 0
        self.last = random.choice([0, 1])
        return self.last

    def step(self, action):
        self.t += 1
        a1, a2, a3, a4, a5 = action
        reward = 0
        if a1 == self.last:
            reward += 1

        # encourage correlation between a1, a2, a3, a4, a5
        if a1 < a2:
            reward += 1
        if a2 < a3:
            reward += 1
        if a3 < a4:
            reward += 1
        if a4 < a5:
            reward += 1

        if a2 < a3 < a4 < a5:
            reward += 5

        done = self.t > 20
        # print("last: {}, a1: {}, a2: {}, a3: {}, a4: {}, a5: {}, reward: {}".format(self.last, a1, a2, a3, a4, a5, reward))
        self.last = random.choice([0, 1])
        return self.last, reward, done, {}


if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    ModelCatalog.register_custom_model("autoregressive_model", AutoregressiveActionsModel)
    ModelCatalog.register_custom_action_dist("autoreg_output", AutoregressiveOutput)
    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config={
            "eager": True,
            "env": CorrelatedCategoricalActionsEnv,
            "gamma": 0.5,
            "lr": 0.0001,
            "num_gpus": 0,
            "model": {
                "custom_model": "autoregressive_model",
                "custom_action_dist": "autoreg_output",
            },
        })
