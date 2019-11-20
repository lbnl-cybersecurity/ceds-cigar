from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml
import argparse
import ray
from ray import tune
from ray.rllib.models.tf.misc import normc_initializer
from gym.spaces import Discrete, Tuple
from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from pycigar.envs.multiagent.wrappers import DISCRETIZE
from ray.rllib.models import ModelCatalog

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=0)

stream = open("../rl_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                  "env_name": "FramestackSingleDiscreteCoopEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ['pv_8', 'pv_9', 'pv_12']}


create_env, env_name = make_create_env(params=pycigar_params, version=0)
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space

tf = try_import_tf()


class SingleOutput(ActionDistribution):
    @staticmethod
    def required_model_output_shape(self, model_config):
        return 32  # controls model output feature vector size

    def sample(self):
        # first, sample a1
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        self._action_logp = a1_dist.logp(a1)

        # return the action tuple
        return a1

    def logp(self, actions):
        a1 = actions
        a1_logits = self.model.action_model([self.inputs])
        return (Categorical(a1_logits).logp(a1))

    def sampled_action_logp(self):
        return tf.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        return a1_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        return a1_terms

    def _a1_distribution(self):
        a1_logits = self.model.action_model([self.inputs])
        a1_dist = Categorical(a1_logits)
        return a1_dist


class SingleActionsModel(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(SingleActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        if action_space != Discrete(DISCRETIZE):
            raise ValueError("This model only supports the [b, b, b, b, b] action space")
        # Inputs
        obs_input = tf.keras.layers.Input(shape=obs_space.shape, name="obs_input")
        ctx_input = tf.keras.layers.Input(shape=(num_outputs, ), name="ctx_input")

        # Output of the model (normally 'logits', but for an autoregressive
        # dist this is more like a context/feature layer encoding the obs)

        #obs_input = obs_input.double()
        conv1d = tf.keras.layers.Conv1D(filters=64,
                                        name="conv1d",
                                        kernel_size=2,
                                        activation=tf.nn.relu)(obs_input)

        flatten = tf.keras.layers.Flatten()(conv1d)
        context = tf.keras.layers.Dense(
            num_outputs,
            name="hidden",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(flatten)

        # V(s)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(context)

        # P(a1 | obs)
        a1_hidden1 = tf.keras.layers.Dense(64, name="a1_hidden1", activation=None, kernel_initializer=normc_initializer(0.01))(ctx_input)
        a1_hidden2 = tf.keras.layers.Dense(32, name="a1_hidden2", activation=None, kernel_initializer=normc_initializer(0.01))(a1_hidden1)
        a1_logits = tf.keras.layers.Dense(DISCRETIZE, name="a1_logits", activation=None, kernel_initializer=normc_initializer(0.01))(a1_hidden2)

        # Base layers
        self.base_model = tf.keras.Model(obs_input, [context, value_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

        # Autoregressive action sampler
        self.action_model = tf.keras.Model(inputs=ctx_input,
                                           outputs=a1_logits)
        self.action_model.summary()
        self.register_variables(self.action_model.variables)

    def forward(self, input_dict, state, seq_lens):
        context, self._value_out = self.base_model(input_dict["obs"])
        return context, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

"""if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config={
            "env": env_name,
            "gamma": 0.5,
            'lr': 5e-03,
            'model': {'conv_filters': None, 'conv_activation': 'relu',
                      'fcnet_activation': 'tanh', 'fcnet_hiddens': [128, 128, 64, 32],
                      'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True,
                      'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256,
                      'lstm_use_prev_action_reward': False, 'state_shape': None,
                      'framestack': False, 'dim': 84, 'grayscale': False,
                      'zero_mean': True, 'custom_preprocessor': None,
                      'custom_model': None, 'custom_action_dist': None, 'custom_options': {}},
            "num_gpus": 0,
            'multiagent': {
                "policies": {
                    "pol": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda x: "pol",
            }
        })
"""


if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    ModelCatalog.register_custom_model("autoregressive_model", SingleActionsModel)
    ModelCatalog.register_custom_action_dist("autoreg_output", SingleOutput)
    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config={
            # "vf_clip_param": 1000.0,
            "env": env_name,
            "gamma": 0.5,
            'eager': True,
            "model": {
                "custom_model": "autoregressive_model",
                "custom_action_dist": "autoreg_output",
            },
            'multiagent': {
                "policies": {
                    "pol": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda x: "pol",
            }
        })
