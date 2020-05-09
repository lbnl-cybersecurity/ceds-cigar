import multiprocessing
import time
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
from pycigar.utils.input_parser import input_parser
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from pycigar.utils.logging import logger
import os
import pycigar

PATH = os.getcwd()


def policy_one(policy, file_name, start=100, percentage_hack=0.45):
    """
    Load the scenarios configuration file. This file contains the scenario information
    for the experiment.
    """
    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, benchmark=True, percentage_hack=percentage_hack)
    pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                      "env_name": "CentralControlPVInverterEnv",
                      "simulator": "opendss"}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)
    sim_params['scenario_config']['start_end_time'] = [start, start + 750]
    del sim_params['attack_randomization']
    test_env = create_env(sim_params)
    test_env.observation_space  # get the observation space, we need this to construct our agent(s) observation input
    test_env.action_space  # get the action space, we need this to construct our agent(s) action output
    tf.compat.v1.enable_eager_execution()
    policy = tf.saved_model.load(policy)
    infer = policy.signatures['serving_default']
    done = False
    obs = test_env.reset()
    obs = obs.tolist()
    while not done:
        act_logits = infer(
            prev_reward=tf.constant([0.], tf.float32),
            observations=tf.constant([obs], tf.float32),
            is_training=tf.constant(False),
            seq_lens=tf.constant([0], tf.int32),
            prev_action=tf.constant([0], tf.int64)
        )['behaviour_logits'].numpy()
        act = np.argmax(act_logits)
        obs, r, done, _ = test_env.step(act)
        obs = obs.tolist()
    log_dict = logger().log_dict
    custom_metrics = logger().custom_metrics

    f = pycigar.utils.output.plot_new(log_dict, custom_metrics)
    f.savefig(os.path.join(PATH, file_name))

if __name__ == '__main__':
    dir_p1 = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar/notebooks/policy_eval_old-obs_M80'

    s = 100
    p_hack = 0.2
    policy_one(policy=dir_p1, file_name="{}-{}p.png".format(s, p_hack), start=s, percentage_hack=p_hack)
    s = 100
    p_hack = 0.45
    policy_one(policy=dir_p1, file_name="{}-{}p.png".format(s, p_hack), start=s, percentage_hack=p_hack)
    s = 11000
    p_hack = 0.2
    policy_one(policy=dir_p1, file_name="{}-{}p.png".format(s, p_hack), start=s, percentage_hack=p_hack)
    s = 11000
    p_hack = 0.45
    policy_one(policy=dir_p1, file_name="{}-{}p.png".format(s, p_hack), start=s, percentage_hack=p_hack)
