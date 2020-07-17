from gym.spaces import Discrete
import numpy as np
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

#tf1, tf, tfv = try_import_tf()
tf = try_import_tf()


class PADICETFModel(TFModelV2):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 actor_hidden_activation="relu",
                 actor_hiddens=(256, 256),
                 critic_hidden_activation="relu",
                 critic_hiddens=(256, 256),
                 twin_q=False,
                 initial_alpha=1.0,
                 target_entropy=None):

        super(PADICETFModel, self).__init__(obs_space, action_space, num_outputs,
                                            model_config, name)

        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.discrete = True
            action_outs = q_outs = self.action_dim
        else:
            self.action_dim = np.product(action_space.shape)
            self.discrete = False
            action_outs = 2 * self.action_dim
            q_outs = 1

        self.model_out = tf.keras.layers.Input(
            shape=(self.num_outputs, ), name="model_out"
        )

        self.action_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=hidden,
                activation=getattr(tf.nn, actor_hidden_activation, None),
                kernel_initializer='orthogonal',
                name="action_{}".format(i + 1))
            for i, hidden in enumerate(actor_hiddens)
        ] + [
            tf.keras.layers.Dense(
                units=action_outs, activation=None, name="action_out",
                kernel_initializer='orthogonal'),
        ])

        self.shift_and_log_scale_diag = self.action_model(self.model_out)
        self.register_variables(self.action_model.variables)

        self.actions_input = None
        if not self.discrete:
            self.actions_input = tf.keras.layers.Input(
                shape=(self.action_dim, ), name="actions")

        def build_q_net(name, observations, actions):
            q_net = tf.keras.Sequential(([
                tf.keras.layers.Concatenate(axis=1),
            ] if not self.discrete else []) + [
                tf.keras.layers.Dense(
                    units=units,
                    activation=getattr(tf.nn, critic_hidden_activation, None),
                    kernel_initializer='orthogonal',
                    name="{}_hidden_{}".format(name, i))
                for i, units in enumerate(critic_hiddens)
            ] + [
                tf.keras.layers.Dense(
                    units=q_outs, activation=None, kernel_initializer='orthogonal', name="{}_out".format(name))
            ])

            if self.discrete:
                q_net = tf.keras.Model(observations, q_net(observations))
            else:
                q_net = tf.keras.Model([observations, actions], q_net([observations, actions]))
            return q_net

        self.q_net = build_q_net("q", self.model_out, self.actions_input)
        self.register_variables(self.q_net.variables)

        if twin_q:
            self.twin_q_net = build_q_net("twin_q", self.model_out, self.actions_input)
            self.register_variables(self.twin_q_net.variables)
        else:
            self.twin_q_net = None

        self.log_alpha = tf.Variable(
            np.log(initial_alpha), dtype=tf.float32, name="log_alpha")
        self.alpha = tf.exp(self.log_alpha)

        # Auto-calculate the target entropy.
        if target_entropy is None or target_entropy == "auto":
            # See hyperparams in [2] (README.md).
            if self.discrete:
                target_entropy = 0.98 * np.array(
                    -np.log(1.0 / action_space.n), dtype=np.float32)
            # See [1] (README.md).
            else:
                target_entropy = -np.prod(action_space.shape)
        self.target_entropy = target_entropy

        self.register_variables([self.log_alpha])

    def get_q_values(self, model_out, actions=None):
        if actions is not None:
            return self.q_net([model_out, actions])
        else:
            return self.q_net(model_out)

    def get_twin_q_values(self, model_out, actions=None):
        if actions is not None:
            return self.twin_q_net([model_out, actions])
        else:
            return self.twin_q_net(model_out)

    def get_policy_output(self, model_output):
        return self.action_model(model_output)

    def policy_variables(self):
        return list(self.action_model.variables)

    def q_variables(self):
        return self.q_net.variables


