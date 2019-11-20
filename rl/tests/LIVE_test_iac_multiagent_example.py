from pycigar.envs.multiagent import MultiIACEnv

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
import argparse
import numpy as np
import ray
import random
from ray import tune
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import Model, ModelCatalog

from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from copy import deepcopy
import gym
from gym.envs.registration import register
from ray.tune import run_experiments
import yaml

"""tf = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-agents", type=int, default=4)
parser.add_argument("--num-policies", type=int, default=2)
parser.add_argument("--num-iters", type=int, default=20)
parser.add_argument("--simple", action="store_true")"""

# time horizon of a single rollout
HORIZON = 1440
# number of rollouts per training iteration
N_ROLLOUTS = 4
# number of parallel workers
N_CPUS = 2

def make_create_env(params, version=0, render=None):
    exp_tag = params["exp_tag"]
    env_name = params["env_name"] + '-v{}'.format(version)
    def create_env(*_):
        sim_params = deepcopy(params['sim_params'])
        env_loc = 'pycigar.envs.multiagent'
        try:
            register(
                id=env_name,
                entry_point=env_loc + ':{}'.format(params["env_name"]),
                kwargs={
                    "sim_params": sim_params,
                    "simulator": params['simulator']
                })
        except Exception:
            pass
        return gym.envs.make(env_name)

    return create_env, env_name


"""class CustomModel1(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Example of (optional) weight sharing between two different policies.
        # Here, we share the variables defined in the 'shared' variable scope
        # by entering it explicitly with tf.AUTO_REUSE. This creates the
        # variables for the 'fc1' layer in a global scope called 'shared'
        # outside of the policy's normal variable scope.
        with tf.variable_scope(
                tf.VariableScope(tf.AUTO_REUSE, "shared"),
                reuse=tf.AUTO_REUSE,
                auxiliary_name_scope=False):
            last_layer = tf.layers.dense(
                input_dict["obs"], 64, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 64, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer


class CustomModel2(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Weights shared with CustomModel1
        with tf.variable_scope(
                tf.VariableScope(tf.AUTO_REUSE, "shared"),
                reuse=tf.AUTO_REUSE,
                auxiliary_name_scope=False):
            last_layer = tf.layers.dense(
                input_dict["obs"], 64, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 64, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer"""


"""if __name__ == "__main__":
    args = parser.parse_args()
    stream = open("pseudo_config_scenarios.yaml", "r")
    sim_params = yaml.safe_load(stream)

    ray.init()

    pycigar_params = {"exp_tag": "exp_1",
                      "env_name": "MultiIACEnv",
                      "sim_params": sim_params,
                      "simulator": "opendss"}

    create_env, gym_name = make_create_env(params=pycigar_params, version=0)
    register_env(gym_name, create_env)
    # Simple environment with `num_agents` independent cartpole entities
    ModelCatalog.register_custom_model("model1", CustomModel1)
    ModelCatalog.register_custom_model("model2", CustomModel2)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": ["model1", "model2"][i % 2],
            },
            "gamma": random.choice([0.95, 0.99]),
        }
        return (None, obs_space, act_space, config)

    # Setup PPO with an ensemble of `num_policies` different policies
    policies = {
        "policy_{}".format(i): gen_policy(i)
        for i in range(args.num_policies)
    }
    policy_ids = list(policies.keys())

    def select_policy(agent_id):
        if agent_id == "pv_8":
            return policy_ids[0]
        else:
            return policy_ids[1]

    tune.run(
        "PPO",
        stop={"training_iteration": args.num_iters},
        config={
            "env": gym_name,
            "log_level": "DEBUG",
            "simple_optimizer": args.simple,
            "num_sgd_iter": 10,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": select_policy,
            },
        },
    )
"""
stream = open("sanity_check_pseudo_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

pycigar_params = {"exp_tag": "exp_1",
                  "env_name": "MultiIACEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss"}

def setup_exps():
    """Return the relevant components of an RLlib experiment.
    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [100, 50, 25]})
    config['use_gae'] = True
    config['lambda'] = 0.97
    config['sgd_minibatch_size'] = 128
    config['kl_target'] = 0.02
    config['num_sgd_iter'] = 10
    config['horizon'] = HORIZON
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config['observation_filter'] = 'NoFilter'

    create_env, env_name = make_create_env(params=pycigar_params, version=0)
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return PPOTFPolicy, obs_space, act_space, {}

    # Setup PG with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'pv_8': gen_policy(), 'pv_9': gen_policy()}

    def policy_mapping_fn(agent_id):
        return agent_id

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': policy_mapping_fn
        }
    })

    return alg_run, env_name, config


if __name__ == '__main__':

    alg_run, env_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS+1)

    run_experiments({
        pycigar_params['exp_tag']: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': 1,
            'stop': {
                'training_iteration': 1000
            },
            'config': config,
        },
    })
