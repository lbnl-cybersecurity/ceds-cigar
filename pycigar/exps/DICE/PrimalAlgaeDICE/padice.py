import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer, calculate_rr_weights
from padice_tf_policy import PADICETFPolicy
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from palice_replay_buffer import PADICELocalReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.execution.replay_ops import StoreToReplayBuffer, Replay
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting

logger = logging.getLogger(__name__)

OPTIMIZER_SHARED_CONFIGS = [
    "buffer_size", "prioritized_replay", "prioritized_replay_alpha",
    "prioritized_replay_beta", "prioritized_replay_eps",
    "rollout_fragment_length", "train_batch_size", "learning_starts"
]

DEFAULT_CONFIG = with_common_config({
    # === Model ===
    "twin_q": False,
    "use_state_preprocessor": False,
    # RLlib model options for the Q function(s).
    "Q_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256, 256],
        "hidden_activation": DEPRECATED_VALUE,
        "hidden_layer_sizes": DEPRECATED_VALUE,
    },
    # RLlib model options for the policy function.
    "policy_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256, 256],
        "hidden_activation": DEPRECATED_VALUE,
        "hidden_layer_sizes": DEPRECATED_VALUE,
    },
    # Unsquash actions to the upper and lower bounds of env's action space.
    # Ignored for discrete action spaces.
    "normalize_actions": True,

    # === Learning ===
    # Disable setting done=True at end of episode. This should be set to True
    # for infinite-horizon MDPs (e.g., many continuous control problems).
    "no_done_at_end": False,
    # Update the target by \tau * policy + (1-\tau) * target_policy.
    "tau": 5e-3,
    # Initial value to use for the entropy weight alpha.
    "initial_alpha": 1.0,
    # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
    # Discrete(2), -3.0 for Box(shape=(3,))).
    # This is the inverse of reward scale, and will be optimized automatically.
    "target_entropy": "auto",
    # N-step target updates.
    "n_step": 1,

    # Number of env steps to optimize for before returning.
    "timesteps_per_iteration": 1000,

    # Alpha in PADICE
    "alpha0": 0.01,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": int(1e6),
    # If True prioritized replay buffer will be used.
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,
    "prioritized_replay_eps": 1e-6,
    "prioritized_replay_beta_annealing_timesteps": 20000,
    "final_prioritized_replay_beta": 0.4,
    # Whether to LZ4 compress observations
    "compress_observations": False,
    # If set, this will fix the ratio of replayed from a buffer and learned on
    # timesteps to sampled from an environment and stored in the replay buffer
    # timesteps. Otherwise, the replay will proceed at the native ratio
    # determined by (train_batch_size / rollout_fragment_length).
    "training_intensity": None,

    # === Optimization ===
    "optimization": {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    },
    # If not None, clip gradients during optimization at this value.
    "grad_clip": None,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 256,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 256,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 1,

    # === Parallelism ===
    # Whether to use a GPU for local optimization.
    "num_gpus": 0,
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Whether to allocate GPUs for workers (if > 0).
    "num_gpus_per_worker": 0,
    # Whether to allocate CPUs for workers (if > 0).
    "num_cpus_per_worker": 1,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span.
    "min_iter_time_s": 1,

    # Whether the loss should be calculated deterministically (w/o the
    # stochastic action sampling step). True only useful for cont. actions and
    # for debugging!
    "_deterministic_loss": False,
    # Use a Beta-distribution instead of a SquashedGaussian for bounded,
    # continuous action spaces (not recommended, for debugging only).
    "_use_beta_distribution": False,

    # DEPRECATED VALUES (set to -1 to indicate they have not been overwritten
    # by user's config). If we don't set them here, we will get an error
    # from the config-key checker.
    "grad_norm_clipping": DEPRECATED_VALUE,
})

def get_policy_class(config):
    return PADICETFPolicy


def validate_config(config):
    if config["model"].get("custom_model"):
        logger.warning(
            "Setting use_state_preprocessor=True since a custom model "
            "was specified.")
        config["use_state_preprocessor"] = True

    if config.get("grad_norm_clipping", DEPRECATED_VALUE) != DEPRECATED_VALUE:
        deprecation_warning("grad_norm_clipping", "grad_clip")
        config["grad_clip"] = config.pop("grad_norm_clipping")

    if config["grad_clip"] is not None and config["grad_clip"] <= 0.0:
        raise ValueError("`grad_clip` value must be > 0.0!")

    # Use same keys as for standard Trainer "model" config.
    for model in ["Q_model", "policy_model"]:
        if config[model].get("hidden_activation", DEPRECATED_VALUE) != \
                DEPRECATED_VALUE:
            deprecation_warning(
                "{}.hidden_activation".format(model),
                "{}.fcnet_activation".format(model),
                error=True)
        if config[model].get("hidden_layer_sizes", DEPRECATED_VALUE) != \
                DEPRECATED_VALUE:
            deprecation_warning(
                "{}.hidden_layer_sizes".format(model),
                "{}.fcnet_hiddens".format(model),
                error=True)

def execution_plan(workers, config):
    if config.get("prioritized_replay"):
        prio_args = {
            "prioritized_replay_alpha": config["prioritized_replay_alpha"],
            "prioritized_replay_beta": config["prioritized_replay_beta"],
            "prioritized_replay_eps": config["prioritized_replay_eps"],
        }
    else:
        prio_args = {}

    local_replay_buffer = PADICELocalReplayBuffer(
        num_shards=1,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        replay_batch_size=config["train_batch_size"],
        #replay_mode=config["multiagent"]["replay_mode"],
        #replay_sequence_length=config["replay_sequence_length"],
        **prio_args)


    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # We execute the following steps concurrently:
    # (1) Generate rollouts and store them in our local replay buffer. Calling
    # next() on store_op drives this.
    store_op = rollouts.for_each(
        StoreToReplayBuffer(local_buffer=local_replay_buffer))

    def update_prio(item):
        samples, info_dict = item
        if config.get("prioritized_replay"):
            prio_dict = {}
            for policy_id, info in info_dict.items():
                # TODO(sven): This is currently structured differently for
                #  torch/tf. Clean up these results/info dicts across
                #  policies (note: fixing this in torch_policy.py will
                #  break e.g. DDPPO!).
                td_error = info.get("td_error",
                                    info[LEARNER_STATS_KEY].get("td_error"))
                prio_dict[policy_id] = (samples.policy_batches[policy_id]
                                        .data.get("batch_indexes"), td_error)
            local_replay_buffer.update_priorities(prio_dict)
        return info_dict

    # (2) Read and train on experiences from the replay buffer. Every batch
    # returned from the LocalReplay() iterator is passed to TrainOneStep to
    # take a SGD step, and then we decide whether to update the target network.
    post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
    replay_op = Replay(local_buffer=local_replay_buffer) \
        .for_each(lambda x: post_fn(x, workers, config)) \
        .for_each(TrainOneStep(workers)) \
        .for_each(update_prio) \
        .for_each(UpdateTargetNetwork(
            workers, config["target_network_update_freq"]))

    # Alternate deterministically between (1) and (2). Only return the output
    # of (2) since training metrics are not available until (2) runs.
    train_op = Concurrently(
        [store_op, replay_op],
        mode="round_robin",
        output_indexes=[1],
        round_robin_weights=calculate_rr_weights(config))

    return StandardMetricsReporting(train_op, workers, config)

PADICETrainer = GenericOffPolicyTrainer.with_updates(
    name="PADICE",
    default_config=DEFAULT_CONFIG,
    default_policy=PADICETFPolicy,
    get_policy_class=get_policy_class,
    validate_config=validate_config,
    execution_plan=execution_plan
)