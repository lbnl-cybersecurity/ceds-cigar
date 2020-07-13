import numpy as np
import random
import os
import collections
import sys

import ray
from ray.rllib.execution.segment_tree import SumSegmentTree, MinSegmentTree
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.util.iter import ParallelIteratorWorker
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.window_stat import WindowStat
from ray.rllib.execution.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class PADICELocalReplayBuffer(ParallelIteratorWorker):
    """A replay buffer shard.
    Ray actors are single-threaded, so for scalability multiple replay actors
    may be created to increase parallelism."""

    def __init__(self,
                 num_shards,
                 learning_starts,
                 buffer_size,
                 replay_batch_size,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 prioritized_replay_eps=1e-6,
                 replay_mode="independent",
                 replay_sequence_length=1,
                 multiagent_sync_replay=False):
        self.replay_starts = learning_starts // num_shards
        self.buffer_size = buffer_size // num_shards
        self.replay_batch_size = replay_batch_size
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_eps = prioritized_replay_eps
        self.multiagent_sync_replay = multiagent_sync_replay

        def gen_replay():
            while True:
                yield self.replay()

        ParallelIteratorWorker.__init__(self, gen_replay, False)

        def new_buffer():
            return PrioritizedReplayBuffer(
                self.buffer_size, alpha=prioritized_replay_alpha)

        def new_init_buffer():
            return ReplayBuffer(self.buffer_size)

        self.replay_buffers = collections.defaultdict(new_buffer)
        self.replay_init_buffers = collections.defaultdict(new_init_buffer)

        # Metrics
        self.add_batch_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.update_priorities_timer = TimerStat()
        self.num_added = 0

        # Make externally accessible for testing.
        global _local_replay_buffer
        _local_replay_buffer = self
        # If set, return this instead of the usual data for testing.
        self._fake_batch = None

    @staticmethod
    def get_instance_for_testing():
        global _local_replay_buffer
        return _local_replay_buffer

    def get_host(self):
        return os.uname()[1]

    def add_batch(self, batch):
        # Make a copy so the replay buffer doesn't pin plasma memory.
        batch = batch.copy()
        # Handle everything as if multiagent
        if isinstance(batch, SampleBatch):
            batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
        with self.add_batch_timer:
            for policy_id, s in batch.policy_batches.items():
                for row in s.rows():
                    self.replay_buffers[policy_id].add(
                        row["obs"], row["actions"], row["rewards"],
                        row["new_obs"], row["dones"], row["weights"]
                        if "weights" in row else None)

            for policy_id, s in batch.policy_batches.items():
                    episodes = s.split_by_episode()
                    for ep in episodes:
                        row = next(ep.rows())
                        self.replay_init_buffers[policy_id].add(
                            row["obs"], row["actions"], row["rewards"],
                            row["new_obs"], row["dones"], row["weights"]
                            if "weights" in row else None)
        self.num_added += batch.count

    def replay(self):
        if self._fake_batch:
            fake_batch = SampleBatch(self._fake_batch)
            return MultiAgentBatch({
                DEFAULT_POLICY_ID: fake_batch
            }, fake_batch.count)

        if self.num_added < self.replay_starts:
            return None

        with self.replay_timer:
            samples = {}
            idxes = None
            for policy_id, replay_buffer in self.replay_buffers.items():
                replay_init_buffer = self.replay_init_buffers[policy_id]
                if self.multiagent_sync_replay:
                    if idxes is None:
                        idxes = replay_buffer.sample_idxes(
                            self.replay_batch_size)
                        idxes_init = replay_init_buffer.sample_idxes(self.replay_batch_size)
                else:
                    idxes = replay_buffer.sample_idxes(self.replay_batch_size)
                    idxes_init = replay_init_buffer.sample_idxes(self.replay_batch_size)
                (obses_t, actions, rewards, obses_tp1, dones, weights,
                 batch_indexes) = replay_buffer.sample_with_idxes(
                     idxes, beta=self.prioritized_replay_beta)

                (obses_0, _, _, _, _) = replay_init_buffer.sample_with_idxes(idxes_init)
                samples[policy_id] = SampleBatch({
                    "obs_0": obses_0,
                    "obs": obses_t,
                    "actions": actions,
                    "rewards": rewards,
                    "new_obs": obses_tp1,
                    "dones": dones,
                    "weights": weights,
                    "batch_indexes": batch_indexes
                })
            return MultiAgentBatch(samples, self.replay_batch_size)

    def update_priorities(self, prio_dict):
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():
                new_priorities = (
                    np.abs(td_errors) + self.prioritized_replay_eps)
                self.replay_buffers[policy_id].update_priorities(
                    batch_indexes, new_priorities)

    def stats(self, debug=False):
        stat = {
            "add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
            "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
            "update_priorities_time_ms": round(
                1000 * self.update_priorities_timer.mean, 3),
        }
        for policy_id, replay_buffer in self.replay_buffers.items():
            stat.update({
                "policy_{}".format(policy_id): replay_buffer.stats(debug=debug)
            })
        return stat


ReplayActor = ray.remote(num_cpus=0)(PADICELocalReplayBuffer)