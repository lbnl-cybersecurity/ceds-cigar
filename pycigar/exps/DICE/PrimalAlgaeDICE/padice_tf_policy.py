from gym.spaces import Box, Discrete
import logging

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.ddpg.ddpg_tf_policy import TargetNetworkMixin
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS, postprocess_nstep_and_prio
from padice_tf_model import PADICETFModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Categorical, Beta, \
                                               DiagGaussian, SquashedGaussian

from ray.rllib.policy.sample_batch import SampleBatch
#from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_tfp
from ray.rllib.utils.tf_ops import make_tf_callable
from padice_dynamic_tf_policy_template import build_tf_policy
# tf1, tf, tfv = try_import_tf()
tf = try_import_tf()
tfp = try_import_tfp()


OBS_0 = "obs_0"

logger = logging.getLogger(__name__)

def build_padice_model(policy, obs_space, action_space, config):
    if config["use_state_preprocessor"]:
        num_outputs = 256
    else:
        num_outputs = 0
        if config["model"]["fcnet_hiddens"]:
            logger.warning(
                "When not using a state-preprocessor with PADICE, `fcnet_hiddens`"
                " will be set to an empty list! Any hidden layer sizes are "
                "defined via `policy_model.fcnet_hiddens` and "
                "`Q_model.fcnet_hiddens`.")
            config["model"]["fcnet_hiddens"] = []

    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface = PADICETFModel,
        name="padice_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"])

    policy.target_model = ModelCatalog.get_model_v2(
    obs_space=obs_space,
    action_space=action_space,
    num_outputs=num_outputs,
    model_config=config["model"],
    framework=config["framework"],
    model_interface = PADICETFModel,
    name="target_padice_model",
    actor_hidden_activation=config["policy_model"]["fcnet_activation"],
    actor_hiddens=config["policy_model"]["fcnet_hiddens"],
    critic_hidden_activation=config["Q_model"]["fcnet_activation"],
    critic_hiddens=config["Q_model"]["fcnet_hiddens"])

    return policy.model

def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    return postprocess_nstep_and_prio(policy, sample_batch)

def get_dist_class(config, action_space):
    if isinstance(action_space, Discrete):
        return Categorical
    else:
        if config["normalize_actions"]:
            return SquashedGaussian if \
                not config["_use_beta_distribution"] else Beta
        else:
            return DiagGaussian

def get_distribution_inputs_and_class(policy,
                                      model,
                                      obs_batch,
                                      *,
                                      explore=True,
                                      **kwargs):
    # Get base-model output.
    model_out, state_out = model({
        "obs": obs_batch,
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)
    # Get action model output from base-model output.
    distribution_inputs = model.get_policy_output(model_out)
    action_dist_class = get_dist_class(policy.config, policy.action_space)
    return distribution_inputs, action_dist_class, state_out

def padice_actor_critic_loss(policy, model, _, train_batch):
    deterministic = policy.config["_deterministic_loss"]

    model_out_t0, _ = model({
        "obs": train_batch[OBS_0],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    model_out_t, _ = model({
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    model_out_tp1, _ = model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    target_model_out_tp1, _ = policy.target_model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    if model.discrete:
        pass
    else:
        action_dist_class = get_dist_class(policy.config, policy.action_space)

        # a0 of current policy given s0
        action_dist_t0 = action_dist_class(
            model.get_policy_output(model_out_t0), policy.model)
        policy_t0 = action_dist_t0.sample() if not deterministic else \
            action_dist_t0.deterministic_sample()
        log_pis_t0 = tf.expand_dims(action_dist_t0.logp(policy_t0), -1)

        # a_tp1 of current policy given s_tp1
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else \
            action_dist_tp1.deterministic_sample()
        log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)


        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])

        # Q-values for the s_tp1, a_tp1
        q_tp1_det_policy = model.get_q_values(model_out_tp1, policy_tp1)

        # Q-values for the s_tp1, a_tp1 of target network
        q_tp1_target_policy = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1)

        # Q-values for the s_t0, a_t0
        q_t0_det_policy = model.get_q_values(model_out_t0, policy_t0)


        r = train_batch[SampleBatch.REWARDS] - model.alpha*log_pis_tp1
        bellman_residual = r + policy.config['gamma']*tf.stop_gradient(q_tp1_target_policy) - q_t
        br_loss = -tf.reduce_mean(tf.math.square(tf.math.maximum(bellman_residual, 0)))
        init_loss = -tf.reduce_mean(2*policy.config['alpha0']*(1-policy.config['gamma'])*q_t0_det_policy)
        alpha_loss = -tf.reduce_mean(model.log_alpha * tf.stop_gradient(log_pis_tp1 + model.target_entropy))

        policy.actor_critic_loss = br_loss + init_loss
        policy.alpha_loss = alpha_loss
        policy.alpha_value = model.alpha
        policy.target_entropy = model.target_entropy
        policy.q_t = q_t

    return br_loss + init_loss + alpha_loss

def gradients_fn(policy, optimizer, loss):
    # Eager: Use GradientTape.

    if policy.config["framework"] in ["tf2", "tfe"]:
        tape = optimizer.tape

        pol_weights = policy.model.policy_variables()
        q_weights = policy.model.q_variables()
        actor_critic_weights = pol_weights + q_weights
        actor_critic_grads_and_vars = list(zip(tape.gradient(
            policy.actor_critic_loss, actor_critic_weights), actor_critic_weights))

        alpha_vars = [policy.model.log_alpha]
        alpha_grads_and_vars = list(zip(tape.gradient(
            policy.alpha_loss, alpha_vars), alpha_vars))
    # Tf1.x: Use optimizer.compute_gradients()
    else:
        pol_weights = policy.model.policy_variables()
        q_weights = policy.model.q_variables()
        actor_critic_weights = pol_weights + q_weights
        actor_critic_grads_and_vars = policy._actor_critic_optimizer.compute_gradients(
            policy.actor_critic_loss, var_list=actor_critic_weights)

        alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
            policy.alpha_loss, var_list=[policy.model.log_alpha])

    # Clip if necessary.
    if policy.config["grad_clip"]:
        clip_func = tf.clip_by_norm
    else:
        clip_func = tf.identity

    # Save grads and vars for later use in `build_apply_op`.
    policy._actor_critic_grads_and_vars = [
        (clip_func(g), v) for (g, v) in actor_critic_grads_and_vars if g is not None]
    policy._alpha_grads_and_vars = [
        (clip_func(g), v) for (g, v) in alpha_grads_and_vars if g is not None]

    grads_and_vars = (
        policy._actor_critic_grads_and_vars + policy._alpha_grads_and_vars)
    return grads_and_vars


def apply_gradients(policy, optimizer, grads_and_vars):
    actor_critic_apply_ops = policy._actor_critic_optimizer.apply_gradients(
        policy._actor_critic_grads_and_vars)

    if policy.config["framework"] in ["tf2", "tfe"]:
        policy._alpha_optimizer.apply_gradients(policy._alpha_grads_and_vars)
        return
    else:
        alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
            policy._alpha_grads_and_vars,
            global_step=tf.train.get_or_create_global_step())
        return tf.group([actor_critic_apply_ops, alpha_apply_ops])

def stats(policy, train_batch):
    return {
    "actor_critic_loss": tf.reduce_mean(policy.actor_critic_loss),
    "alpha_loss": tf.reduce_mean(policy.alpha_loss),
    "alpha_value": tf.reduce_mean(policy.alpha_value),
    "target_entropy": tf.constant(policy.target_entropy),
    "mean_q": tf.reduce_mean(policy.q_t),
    "max_q": tf.reduce_max(policy.q_t),
    "min_q": tf.reduce_min(policy.q_t)
    }

class ActorCriticOptimizerMixin:
    def __init__(self, config):
        # - Create global step for counting the number of update operations.
        # - Use separate optimizers for actor & critic.
        if config["framework"] in ["tf2", "tfe"]:
            self.global_step = get_variable(0, tf_name="global_step")
            self._actor_critic_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["actor_critic_learning_rate"])
            self._alpha_optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimization"]["entropy_learning_rate"])
        else:
            self.global_step = tf.train.get_or_create_global_step()
            self._actor_critic_optimizer = tf.train.AdamOptimizer(
                learning_rate=config["optimization"]["actor_critic_learning_rate"])
            self._alpha_optimizer = tf.train.AdamOptimizer(
                learning_rate=config["optimization"]["entropy_learning_rate"])

class ComputePADICEErrorMixin:
    def __init__(self, loss_fn):
        @make_tf_callable(self.get_session(), dynamic_shape=True)
        def compute_td_error(obs_0, obs_t, act_t, rew_t, obs_tp1, done_mask,
                             importance_weights):
            # Do forward pass on loss to update td errors attribute
            # (one TD-error value per item in batch to update PR weights).
            loss_fn(
                self, self.model, None, {
                    OBS_0: tf.convert_to_tensor(obs_0),
                    SampleBatch.CUR_OBS: tf.convert_to_tensor(obs_t),
                    SampleBatch.ACTIONS: tf.convert_to_tensor(act_t),
                    SampleBatch.REWARDS: tf.convert_to_tensor(rew_t),
                    SampleBatch.NEXT_OBS: tf.convert_to_tensor(obs_tp1),
                    SampleBatch.DONES: tf.convert_to_tensor(done_mask),
                    PRIO_WEIGHTS: tf.convert_to_tensor(importance_weights),
                })
            # `self.td_error` is set in loss_fn.
            return self.td_error

        self.compute_td_error = compute_td_error

def setup_early_mixins(policy, obs_space, action_space, config):
    ActorCriticOptimizerMixin.__init__(policy, config)


def setup_mid_mixins(policy, obs_space, action_space, config):
    ComputePADICEErrorMixin.__init__(policy, padice_actor_critic_loss)


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, config)


def validate_spaces(pid, observation_space, action_space, config):
    if not isinstance(action_space, (Box, Discrete)):
        raise UnsupportedSpaceException(
            "Action space ({}) of {} is not supported for "
            "SAC.".format(action_space, pid))
    if isinstance(action_space, Box) and len(action_space.shape) > 1:
        raise UnsupportedSpaceException(
            "Action space ({}) of {} has multiple dimensions "
            "{}. ".format(action_space, pid, action_space.shape) +
            "Consider reshaping this into a single dimension, "
            "using a Tuple action space, or the multi-agent API.")


PADICETFPolicy = build_tf_policy(
    name="PADICETFPolicy",
    get_default_config=lambda: ray.rllib.agents.sac.sac.DEFAULT_CONFIG,
    make_model=build_padice_model,
    postprocess_fn=None,
    action_distribution_fn=get_distribution_inputs_and_class,
    loss_fn=padice_actor_critic_loss,
    stats_fn=stats,
    gradients_fn=gradients_fn,
    apply_gradients_fn=apply_gradients,
    #extra_learn_fetches_fn=lambda policy: {"td_error": policy.td_error},
    mixins=[
        TargetNetworkMixin, ActorCriticOptimizerMixin, ComputePADICEErrorMixin
    ],
    #validate_spaces=validate_spaces,
    before_init=setup_early_mixins,
    before_loss_init=setup_mid_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False)