from gym.spaces import Box

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
COOP_OBS = "coop_obs"
COOP_ACTION = "coop_action"

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

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        obs = tf.keras.layers.Input(shape=obs_space.shape, name="obs")

        other_agent_obs_acts = []
        for i in range(model_config['custom_model_config']['num_agents'] - 1):
            other_agent_obs_acts.append(tf.keras.layers.Input(shape=obs_space.shape, name="coop_obs_{}".format(i)))
            other_agent_obs_acts.append(tf.keras.layers.Input(shape=21*3, name="coop_act_{}".format(i)))

        concat_obs = tf.keras.layers.Concatenate(axis=1)(
            [obs, *other_agent_obs_acts])
        central_vf_dense = tf.keras.layers.Dense(
            16, activation=tf.nn.tanh, name="c_vf_dense")(concat_obs)
        central_vf_out = tf.keras.layers.Dense(
            1, activation=None, name="c_vf_out")(central_vf_dense)

        self.central_vf = tf.keras.Model(
            inputs=[obs, *other_agent_obs_acts], outputs=central_vf_out)

        self.register_variables(self.central_vf.variables)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def central_value_function(self, obs, *args):
        coop = []
        for coop_value in args:
            if coop_value.shape[1] != 3:
                coop.append(coop_value)
            else:
                coop.append(tf.reshape(tf.one_hot(coop_value, 21), [-1, 21*3]))
        return tf.reshape(self.central_vf([obs, *coop]), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used


class YetAnotherCentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(YetAnotherCentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        self.action_model = FullyConnectedNetwork(
            Box(low=0, high=1, shape=(6, )),  # one-hot encoded Discrete(6)
            action_space,
            num_outputs,
            model_config,
            name + "_action")
        self.register_variables(self.action_model.variables())

        self.value_model = FullyConnectedNetwork(obs_space, action_space, 1,
                                                 model_config, name + "_vf")
        self.register_variables(self.value_model.variables())

    def forward(self, input_dict, state, seq_lens):
        self._value_out, _ = self.value_model({
            "obs": input_dict["obs_flat"]
        }, state, seq_lens)
        return self.action_model({
            "obs": input_dict["obs"]["own_obs"]
        }, state, seq_lens)

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
