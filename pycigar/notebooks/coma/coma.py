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
from coma_model import \
    CentralizedCriticModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable


tf1, tf, tfv = try_import_tf()

COOP_OBS = "coop_obs"
COOP_ACTION = "coop_action"

parser = argparse.ArgumentParser()
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=7.99)


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = make_tf_callable(self.get_session())(self.model.central_value_function)


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
            #coop_obs_acts.append(sample_batch[COOP_ACTION + str(i)])
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(sample_batch[SampleBatch.CUR_OBS], *coop_obs_acts)
    else:
        # Policy hasn't been initialized yet, use zeros.
        num_other_agents = len(policy.config['multiagent']['policies'])-1
        for idx in range(num_other_agents):
            sample_batch[COOP_OBS + str(idx)] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
            #sample_batch[COOP_ACTION + str(idx)] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = tf_loss

    vf_saved = model.value_function
    num_other_agents = len(policy.config['multiagent']['policies'])-1
    coop_obs_acts = []
    for i in range(num_other_agents):
        coop_obs_acts.append(train_batch[COOP_OBS + str(i)])
        #coop_obs_acts.append(train_batch[COOP_ACTION + str(i)])
    model.value_function = lambda: policy.model.central_value_function(train_batch[SampleBatch.CUR_OBS], *coop_obs_acts)

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


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
    grad_stats_fn=central_vf_stats,
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
