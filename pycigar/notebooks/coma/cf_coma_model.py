from gym.spaces import Box

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
COOP_OBS = "coop_obs"
COOP_ACTION = "coop_action"
tf.compat.v1.disable_eager_execution()

class CentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized value function."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        # Base of the model
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

        with tf1.variable_scope('central_critic/', reuse=tf1.AUTO_REUSE) as scope:
        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
            obs = tf.keras.layers.Input(shape=obs_space.shape, name="obs")

            other_agent_obs_acts = []
            for i in range(model_config['custom_model_config']['num_agents'] - 1):
                other_agent_obs_acts.append(tf.keras.layers.Input(shape=obs_space.shape, name="coop_obs_{}".format(i)))
                other_agent_obs_acts.append(tf.keras.layers.Input(shape=21*3, name="coop_act_{}".format(i)))

            concat_obs = tf.keras.layers.Concatenate(axis=1)([obs, *other_agent_obs_acts])
            central_q_dense = tf.keras.layers.Dense(32, activation=tf.nn.tanh, name="c_q_dense_0")(concat_obs)
            central_q_dense = tf.keras.layers.Dense(32, activation=tf.nn.tanh, name="c_q_dense_1")(central_q_dense)
            central_q_out = tf.keras.layers.Dense(21**3, activation=None, name="c_q_out")(central_q_dense)
            self.central_q = tf.keras.Model(inputs=[obs, *other_agent_obs_acts], outputs=central_q_out)


            self.register_variables(self.central_q.variables)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    """def central_value_function(self, obs, *args):
        coop = []
        for coop_value in args:
            if coop_value.shape[1] != 3:
                coop.append(coop_value)
            else:
                coop.append(tf.reshape(tf.one_hot(coop_value, 21), [-1, 21*3]))
        return tf.reshape(self.central_vf([obs, *coop]), [-1])"""

    def central_q_function(self, obs, *args):
        coop = []
        for coop_value in args:
            if coop_value.shape[1] != 3:
                coop.append(coop_value)
            else:
                coop.append(tf.reshape(tf.one_hot(coop_value, 21), [-1, 21*3]))
        return  self.central_q([obs, *coop])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used

