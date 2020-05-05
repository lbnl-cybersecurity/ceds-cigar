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


def policy_one(policy, file_name):
    """
    Load the scenarios configuration file. This file contains the scenario information
    for the experiment.
    """
    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, benchmark=True)
    pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                      "env_name": "CentralControlPVInverterEnv",
                      "simulator": "opendss"}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)
    sim_params['scenario_config']['start_end_time'] = [100, 100 + 750]
    del sim_params['attack_randomization']
    test_env = create_env(sim_params)
    test_env.observation_space  # get the observation space, we need this to construct our agent(s) observation input
    test_env.action_space  # get the action space, we need this to construct our agent(s) action output
    tf.compat.v1.enable_eager_execution()
    policy = tf.saved_model.load(policy)
    infer = policy.signatures['serving_default']
    done = False
    reward = 0
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
        reward += r
        log_dict = logger().log_dict
        logger().log('reward', 'reward', reward)
        pickle.dump(log_dict, open(os.path.join(PATH, "action.pkl"), "wb"))
        pickle.dump(logger().custom_metrics, open(os.path.join(PATH, "custom_metrics.pkl"), "wb"))


def policy_two(policy, file_name):
    misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
    dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
    load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
    breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

    sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path, benchmark=True)
    pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                      "env_name": "CentralControlPVInverterEnv",
                      "simulator": "opendss"}

    create_env, env_name = make_create_env(pycigar_params, version=0)
    register_env(env_name, create_env)
    sim_params['scenario_config']['start_end_time'] = [100, 100 + 750]
    del sim_params['attack_randomization']
    test_env = create_env(sim_params)
    test_env.observation_space  # get the observation space, we need this to construct our agent(s) observation input
    test_env.action_space  # get the action space, we need this to construct our agent(s) action output
    custom_metrics = pickle.load(open(os.path.join(PATH, "custom_metrics.pkl"), "rb"))
    randomize_rl_update = custom_metrics['randomize_rl_update']
    tf.compat.v1.enable_eager_execution()
    policy = tf.saved_model.load(policy)
    infer = policy.signatures['serving_default']
    done = False
    reward = 0
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
        obs, r, done, _ = test_env.step(act, randomize_rl_update.pop(0))
        obs = obs.tolist()
        reward += r
        log_dict = logger().log_dict
        logger().log('reward', 'reward', reward)
        pickle.dump(log_dict, open(os.path.join(PATH, "new_action.pkl"), "wb"))


def plot(policy, file_name):
    log_dict = pickle.load(open(os.path.join(PATH, "action.pkl"), "rb"))
    log_dict_new = pickle.load(open(os.path.join(PATH, "new_action.pkl"), "rb"))

    tracking_ids = ['inverter_s701a', 'inverter_s701b', 'inverter_s701c',
                    'inverter_s712c', 'inverter_s713c', 'inverter_s714a',
                    'inverter_s714b', 'inverter_s718a', 'inverter_s720c',
                    'inverter_s722b', 'inverter_s722c', 'inverter_s724b',
                    'inverter_s725b', 'inverter_s727c', 'inverter_s728',
                    'inverter_s729a', 'inverter_s730c', 'inverter_s731b',
                    'inverter_s732c', 'inverter_s733a', 'inverter_s734c',
                    'inverter_s735c', 'inverter_s736b', 'inverter_s737a',
                    'inverter_s738a', 'inverter_s740c', 'inverter_s741c',
                    'inverter_s742a', 'inverter_s742b', 'inverter_s744a']

    hack_tracking_ids = ['adversary_inverter_s701a', 'adversary_inverter_s701b', 'adversary_inverter_s701c',
                         'adversary_inverter_s712c', 'adversary_inverter_s713c', 'adversary_inverter_s714a',
                         'adversary_inverter_s714b', 'adversary_inverter_s718a', 'adversary_inverter_s720c',
                         'adversary_inverter_s722b', 'adversary_inverter_s722c', 'adversary_inverter_s724b',
                         'adversary_inverter_s725b', 'adversary_inverter_s727c', 'adversary_inverter_s728',
                         'adversary_inverter_s729a', 'adversary_inverter_s730c', 'adversary_inverter_s731b',
                         'adversary_inverter_s732c', 'adversary_inverter_s733a', 'adversary_inverter_s734c',
                         'adversary_inverter_s735c', 'adversary_inverter_s736b', 'adversary_inverter_s737a',
                         'adversary_inverter_s738a', 'adversary_inverter_s740c', 'adversary_inverter_s741c',
                         'adversary_inverter_s742a', 'adversary_inverter_s742b', 'adversary_inverter_s744a']

    def translation(log_dict, tracking_id):
        return (np.array(log_dict[tracking_id]['control_setting']) - np.array(log_dict[tracking_id]['control_setting'])[0, 2])[:, 2]

    def slope(log_dict, tracking_id):
        return np.array(log_dict[tracking_id]['control_setting'])[:, 1] - np.array(log_dict[tracking_id]['control_setting'])[:, 0]

    f, ax = plt.subplots(7, figsize=(25, 25))
    tracking_id = 'inverter_s701a'
    node = 's701a'
    f.suptitle('reward old:{}. reward new:{}'.format(log_dict['reward']['reward'], log_dict_new['reward']['reward']), fontsize=20)
    [v_old, v_new] = ax[0].plot(np.array(list(zip(log_dict[node]['voltage'], log_dict_new[node]['voltage']))))
    ax[0].legend([v_old, v_new], ['v_old', 'v_new'], loc=1)
    ax[0].set_ylabel('voltage')
    ax[0].set_ylim((0.97, 1.04))
    ax[0].grid(b=True, which='both')

    [y_old, y_new] = ax[1].plot(np.array(list(zip(log_dict[tracking_id]['y'], log_dict_new[tracking_id]['y']))))
    ax[1].legend([y_old, y_new], ['y_old', 'y_new'], loc=1)
    ax[1].set_ylabel('oscillation observer')
    ax[1].grid(b=True, which='both')
    ax[2].plot(log_dict[tracking_id]['q_set'])
    ax[2].plot(log_dict[tracking_id]['q_out'])
    ax[2].set_ylabel('reactive power')
    ax[2].grid(b=True, which='both')
    labels = ['a1', 'a2']
    for tracking_id in tracking_ids:
        [a1, a2] = ax[3].plot(np.array(list(zip(slope(log_dict, tracking_id), translation(log_dict, tracking_id)))))
    ax[3].set_ylabel('action')
    ax[3].grid(b=True, which='both')

    for tracking_id in tracking_ids:
        [a1, a2] = ax[3].plot(np.array(list(zip(slope(log_dict_new, tracking_id), translation(log_dict_new, tracking_id)))))
    ax[3].set_ylabel('action')
    ax[3].grid(b=True, which='both')
    ax[3].legend([a1, a2], labels, loc=1)

    for tracking_id in hack_tracking_ids:
        [a1, a2] = ax[4].plot(np.array(list(zip(slope(log_dict, tracking_id), translation(log_dict, tracking_id)))), color='blue')
    ax[4].set_ylabel('action')
    ax[4].grid(b=True, which='both')
    ax[4].legend([a1, a2], labels, loc=1)

    for tracking_id in hack_tracking_ids:
        [a1, a2] = ax[4].plot(np.array(list(zip(slope(log_dict_new, tracking_id), translation(log_dict_new, tracking_id)))), color='orange')
    ax[4].set_ylabel('action')
    ax[4].grid(b=True, which='both')
    ax[4].legend([a1, a2], labels, loc=1)

    substation_p = np.array(log_dict['network']['substation_power'])[:, 0]
    substation_q = np.array(log_dict['network']['substation_power'])[:, 1]
    substation_p_new = np.array(log_dict_new['network']['substation_power'])[:, 0]
    substation_q_new = np.array(log_dict_new['network']['substation_power'])[:, 1]
    [substation_p, substation_p_new] = ax[5].plot(np.array(list(zip(substation_p, substation_p_new))))
    ax[5].legend([substation_p, substation_p_new], ['substation_p_old', 'substation_p_new'], loc=1)
    ax[5].set_ylabel('substation power')
    ax[5].grid(b=True, which='both')
    [substation_q, substation_q_new] = ax[6].plot(np.array(list(zip(substation_q, substation_q_new))))
    ax[6].legend([substation_q, substation_q_new], ['substation_q_old', 'substation_q_new'], loc=1)
    ax[6].set_ylabel('substation reactive power')
    ax[6].grid(b=True, which='both')

    ax[0].set_xlim(180, 500)
    ax[1].set_xlim(180, 500)
    ax[2].set_xlim(180, 500)
    ax[3].set_xlim(180, 500)
    ax[4].set_xlim(180, 500)
    ax[5].set_xlim(180, 500)

    f.savefig(os.path.join(PATH, file_name))


def clean_up(policy, file_name):
    file_names = os.listdir(PATH)
    file_names_pkl = [x for x in file_names if '.pkl' in x]
    for filename in file_names_pkl:
        os.remove(filename)


class Process(multiprocessing.Process):
    def __init__(self, id, func, policy=None, file_name=None):
        super(Process, self).__init__()
        self.id = id
        self.func = func
        self.policy = policy
        self.file_name = file_name

    def run(self):
        time.sleep(1)
        print('Begin process {}'.format(self.id))
        self.func(self.policy, self.file_name)
        print('Done process {}'.format(self.id))


if __name__ == '__main__':
    dir_p1 = '/home/toanngo/hp_experiment3/main_old_action/run_train/run_train_e5738110_0_2020-04-17_05-42-2850clbrq3/best/policy'
    dir_p2 = '/home/toanngo/hp_experiment3/main/run_train/run_train_aab140ae_0_2020-04-16_19-25-120ftkcvfz/best/policy'

    p = Process(0, policy_one, policy=dir_p1)
    p.start()
    p.join()
    p = Process(1, policy_two, policy=dir_p1)
    p.start()
    p.join()
    p = Process(2, plot, file_name='result.png')
    p.start()
    p.join()

    p = Process(3, clean_up)
    p.start()
    p.join()
