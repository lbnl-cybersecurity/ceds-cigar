"""An example of customizing PPO to leverage a centralized critic.
Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.
Compared to simply running `rllib/examples/two_step_game.py --run=PPO`,
this centralized critic version reaches vf_explained_variance=1.0 more stably
since it takes into account the opponent actions as well as the policy's.
Note that this is also using two independent policies instead of weight-sharing
with one.
See also: centralized_critic_2.py for a simpler approach that instead
modifies the environment.
"""

import argparse
import numpy as np
from gym.spaces import Discrete
import os

import ray
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
from cf_coma_model import CentralizedCriticModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
import itertools
import scipy

tf1, tf, tfv = try_import_tf()

COOP_OBS = "coop_obs"
COOP_ACTION = "coop_action"

parser = argparse.ArgumentParser()
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=7.99)

mapping_action_q = np.array(list(itertools.product(range(21), range(21), range(21))))
def discount(x: np.ndarray, gamma: float):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        #self.compute_central_vf = make_tf_callable(self.get_session())(self.model.central_value_function)
        self.compute_central_q = make_tf_callable(self.get_session())(self.model.central_q_function)


def compute_advantages_coma(rollout,
                            last_r,
                            gamma=0.9,
                            lambda_=1.0,
                            use_gae=False,
                            use_critic=True):
    """
    Given a rollout, compute its value targets and the advantages.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory.
        last_r (float): Value estimation for last observation.
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE.
        use_gae (bool): Using Generalized Advantage Estimation.
        use_critic (bool): Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    rollout_size = len(rollout[SampleBatch.ACTIONS])

    """assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (
            rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        rollout[Postprocessing.VALUE_TARGETS] = (
            rollout[Postprocessing.ADVANTAGES] +
            rollout[SampleBatch.VF_PREDS]).copy().astype(np.float32)
    else:"""
    rewards_plus_v = np.concatenate(
        [rollout[SampleBatch.REWARDS],
            np.array([last_r])])
    discounted_returns = discount(rewards_plus_v,
                                    gamma)[:-1].copy().astype(np.float32)

    if use_critic:
        rollout[Postprocessing.
                ADVANTAGES] = discounted_returns - rollout['q_baseline']
        rollout[Postprocessing.VALUE_TARGETS] = discounted_returns
    else:
        rollout[Postprocessing.ADVANTAGES] = discounted_returns
        rollout[Postprocessing.VALUE_TARGETS] = np.zeros_like(
            rollout[Postprocessing.ADVANTAGES])

    rollout[Postprocessing.ADVANTAGES] = rollout[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == rollout_size for key, val in rollout.items()), \
        "Rollout stacked incorrectly!"
    return rollout

# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    if policy.loss_initialized():
        assert other_agent_batches is not None

        num_other_agents = len(other_agent_batches)
        for idx, agent_id in enumerate(other_agent_batches):
            (_, other_batch) = other_agent_batches[agent_id]

            # also record the opponent obs and actions in the trajectory
            sample_batch[COOP_OBS + str(idx)] = other_batch[SampleBatch.CUR_OBS]
            sample_batch[COOP_ACTION + str(idx)] = other_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        coop_obs_acts = []
        for i in range(num_other_agents):
            coop_obs_acts.append(sample_batch[COOP_OBS + str(i)])
            coop_obs_acts.append(sample_batch[COOP_ACTION + str(i)])
        #sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(sample_batch[SampleBatch.CUR_OBS], *coop_obs_acts)
        sample_batch['q_all_action'] = policy.compute_central_q(sample_batch[SampleBatch.CUR_OBS], *coop_obs_acts)
        sample_batch['q_mask'] = np.zeros_like(sample_batch['q_all_action'], dtype=np.float32)
        sample_batch['q_baseline'] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)
        logits = np.reshape(sample_batch['action_dist_inputs'], [sample_batch['action_dist_inputs'].shape[0], 3, 21])
        action_probs = np.exp(logits)/np.expand_dims(np.sum(np.exp(logits), axis=2), axis=-1)
        all_action_probs = np.zeros_like(sample_batch['action_dist_inputs'], dtype=np.float32)
        for i, a in enumerate(sample_batch[SampleBatch.ACTIONS]):
            pos = np.where(np.all(mapping_action_q == a, axis=1))[0][0]
            sample_batch['q_mask'][i][pos] = 1
            q_val = sample_batch['q_all_action'][i][pos]

            #other_q_baseline = 0
            #for other_a in mapping_action_q:
            #    pos = np.where(np.all(mapping_action_q == a, axis=1))[0][0]
            #    other_a_prob = action_probs[i, 0, other_a[0]] * action_probs[i, 1, other_a[1]] * action_probs[i, 2, other_a[2]]
            #    other_q_baseline += other_a_prob * sample_batch['q_all_action'][i][pos]
            #sample_batch['q_baseline'][i] = -other_q_baseline
        #logits = sample_batch['action_dist_inputs']
        #action_probs = np.exp(logits) / np.expand_dims(np.sum(np.exp(logits), axis=1), axis=1)
        #sample_batch['cf_baseline'] = np.sum(sample_batch['cf_baseline'] * action_probs, axis=1) # not right, need to excude the Q of action
            probs = np.outer(np.outer(action_probs[i, 0, :], action_probs[i, 1, :]), action_probs[i, 2, :]).flatten()
            sample_batch['q_baseline'][i] = np.sum(np.multiply(probs, sample_batch['q_all_action'][0])) - q_val * probs[pos]

        sample_batch[SampleBatch.VF_PREDS] = np.sum(sample_batch['q_all_action'] * sample_batch['q_mask'], axis=-1)
    else:
        # Policy hasn't been initialized yet, use zeros.
        num_other_agents = len(policy.config['multiagent']['policies'])-1
        for idx in range(num_other_agents):
            sample_batch[COOP_OBS + str(idx)] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
            sample_batch[COOP_ACTION + str(idx)] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)
        sample_batch['q_all_action'] = np.zeros((1, 21**3), dtype=np.float32)
        sample_batch['q_mask'] = np.zeros_like(sample_batch['q_all_action'], dtype=np.float32)
        sample_batch['q_baseline'] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)
        #sample_batch['q_position'] = np.zeros((1, 21**3), dtype=np.float32)
    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages_coma(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


# TODO: change loss function
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = ppo_surrogate_loss_coma

    num_other_agents = len(policy.config['multiagent']['policies'])-1
    coop_obs_acts = []
    for i in range(num_other_agents):
        coop_obs_acts.append(train_batch[COOP_OBS + str(i)])
        coop_obs_acts.append(train_batch[COOP_ACTION + str(i)])
    #model.value_function = lambda: policy.model.central_value_function(train_batch[SampleBatch.CUR_OBS], *coop_obs_acts)
    model.q_function = lambda: policy.model.central_q_function(train_batch[SampleBatch.CUR_OBS], *coop_obs_acts)
    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    return loss

def ppo_surrogate_loss_coma(policy, model, dist_class, train_batch):
    """Constructs the loss for Proximal Policy Objective.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    logits, state = model.from_batch(train_batch)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, mask))

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = tf.reduce_mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)

    logp_ratio = tf.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP])
    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = tf.minimum(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES] * tf.clip_by_value(
            logp_ratio, 1 - policy.config["clip_param"],
            1 + policy.config["clip_param"]))
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    if policy.config["use_gae"]:
        prev_q_fn_out = train_batch[SampleBatch.VF_PREDS]
        q_fn_out = tf.reduce_sum(model.q_function()*train_batch['q_mask'], -1)
        qf_loss1 = tf.math.square(q_fn_out -
                                  train_batch[Postprocessing.VALUE_TARGETS])
        qf_clipped = prev_q_fn_out + tf.clip_by_value(
            q_fn_out - prev_q_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        qf_loss2 = tf.math.square(qf_clipped -
                                  train_batch[Postprocessing.VALUE_TARGETS])
        qf_loss = tf.maximum(qf_loss1, qf_loss2)
        mean_qf_loss = reduce_mean_valid(qf_loss)
        #prev_q_fn_out = train_batch[SampleBatch.VF_PREDS]
        #qf_loss1 = tf.math.square(q_fn_out - train_batch[Postprocessing.VALUE_TARGETS])
        total_loss = reduce_mean_valid(
            -surrogate_loss + policy.kl_coeff * action_kl +
            policy.config["vf_loss_coeff"] * qf_loss -
            policy.entropy_coeff * curr_entropy)
    else:
        mean_vf_loss = tf.constant(0.0)
        total_loss = reduce_mean_valid(-surrogate_loss +
                                       policy.kl_coeff * action_kl -
                                       policy.entropy_coeff * curr_entropy)

    # Store stats in policy for stats_fn.
    policy._total_loss = total_loss
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_vf_loss = mean_qf_loss
    policy._mean_entropy = mean_entropy
    policy._mean_kl = mean_kl

    return total_loss

def setup_tf_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out),
    }


CCPPOTFPolicy = PPOTFPolicy.with_updates(
    name="CCPPOTFPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_tf_mixins,
    #grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])


CCTrainer = PPOTrainer.with_updates(
    name="CCPPOTrainer",
    default_policy=CCPPOTFPolicy
)

if __name__ == "__main__":
    ray.init(local_mode=True)
    args = parser.parse_args()

    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    config = {
        "env": TwoStepGame,
        "batch_mode": "complete_episodes",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "multiagent": {
            "policies": {
                "pol1": (None, Discrete(6), TwoStepGame.action_space, {}),
                "pol2": (None, Discrete(6), TwoStepGame.action_space, {}),
            },
            "policy_mapping_fn": lambda x: "pol1" if x == 0 else "pol2",
        },
        "model": {
            "custom_model": "cc_model"
        },
        "framework": "tf",
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(CCTrainer, config=config, stop=stop, verbose=1)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
