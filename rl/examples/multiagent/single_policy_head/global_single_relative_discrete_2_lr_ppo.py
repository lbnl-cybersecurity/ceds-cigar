import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
import argparse
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork 
from gym.spaces import Box, Dict, Discrete
from ray.rllib.utils import try_import_tf
import numpy as np 

from pycigar.envs.multiagent.wrappers import DISCRETIZE_RELATIVE

SAVE_RATE = 5

tf = try_import_tf()


"""
Load the scenarios configuration file. This file contains the scenario information
for the experiment.
"""
stream = open("../rl_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

"""
Register the environment to OpenGymAI. This is necessary, RLlib can find the new environment
with string name env_name_v:version:, ex. SingleRelativeDiscreteCoopEnv_v0.
env_name: name of environment being used.
sim_params: simulation params, it is the scenario configuration.
simulator: the simulator being used, ex. opendss, gridlabd...
tracking_ids: list of ids of devices being tracked during the experiment.
"""

pycigar_params = {"exp_tag": "cooperative_multiagent_ppo",
                  "env_name": "SingleRelativeDiscreteCoopEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ['pv_17']}
"""
call function make_create_env() to register the new environment to OpenGymAI.
create_env() is a function to create new instance of the environment.
env_name: the registered name of the new environment.
"""
create_env, env_name = make_create_env(params=pycigar_params, version=0)
register_env(env_name, create_env)
test_env = create_env()
obs_space = test_env.observation_space  # get the observation space, we need this to construct our agent(s) observation input
act_space = test_env.action_space  # get the action space, we need this to construct our agent(s) action output


"""
Define training process. Ray/Tune will call this function with config params.
"""


def testspeed_ppo_coop_train(config, reporter):

    # initialize PPO agent on environment. This may create 1 or more workers.
    agent1 = PPOTrainer(env=env_name, config=config)
    #agent1.restore('/home/toanngo/ray_results/checkpoint_ngae/checkpoint_35/checkpoint-35')
    # begin train iteration
    for i in range(5000):
        # this function will collect samples and train agent with the samples.
        # result is the summerization of training progress.
        result = agent1.train()
        # adding a phase into the result dictionary.
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]

        # for every SAVE_RATE training iterations, we test the agent on the test environment to see how it performs.
        if i != 0 and (i+1) % SAVE_RATE == 0:
            state = agent1.save('~/ray_results/checkpoint')
            done = False
            # reset the test environment
            obs = test_env.reset()
            while not done:
                act = {}
                for k, v in obs.items():
                    # for each observation, let the policy decides what to do
                    act[k] = agent1.compute_action(v, policy_id='pol')
                # forward 1 step with agent action
                obs, _, done, _ = test_env.step(act)
                done = done['__all__']
            # plot the result. This will be saved in ./results
            test_env.plot(pycigar_params['exp_tag'], env_name, i+1)

    # save the params of agent
    # state = agent1.save('/home/toanngo/ray_results/checkpoint')
    # stop the agent
    agent1.stop()

def fill_in_actions(info):
    """Callback that saves opponent actions into the agent obs.
    If you don't care about opponent actions you can leave this out."""

    to_update = info["post_batch"][SampleBatch.CUR_OBS]
    my_id = info["agent_id"]
    other_ids = list(info["all_pre_batches"].keys())
    other_ids.remove(my_id)


    action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(DISCRETIZE_RELATIVE))

    # set the opponent actions into the observation
    opponent_actions = []
    for other_id in other_ids:
        _, opponent_batch = info["all_pre_batches"][other_id]
        if opponent_actions == []:
            opponent_actions = np.array([action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]])
        else:
            new_opponent_actions = np.array([action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]])
            opponent_actions = np.concatenate((opponent_actions, new_opponent_actions), axis=1)
    
    to_update[:, :opponent_actions.shape[1]] = opponent_actions


class CentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized VF.
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
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        self.action_model = FullyConnectedNetwork(
            Box(low=-float('inf'), high=float('inf'), shape=(int(obs_space.shape[0]/12-DISCRETIZE_RELATIVE), )),  # own_obs
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

if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
    #config for RLlib - PPO agent
    config = {
        "gamma": 0.5,
        'lr': 5e-04,
        'sample_batch_size': 50,
        #'lr_schedule': [[0, 5e-04], [50000, 5e-05]],
        "batch_mode": "complete_episodes",
        "callbacks": {
            "on_postprocess_traj": fill_in_actions,
        },
        #worker
        'num_workers': 2,
        'num_gpus': 0,
        'num_cpus_per_worker': 1,
        'num_gpus_per_worker': 0,
        'custom_resources_per_worker': {},
        'num_cpus_for_driver': 1,
        'memory': 0,
        'object_store_memory': 0,
        'memory_per_worker': 0,
        'object_store_memory_per_worker': 0,
        'num_envs_per_worker': 1,
        'evaluation_num_episodes': 1,
        #model
        'model': {
            'fcnet_hiddens': [256, 256, 128, 64, 32],
            "custom_model": "cc_model",
            },
        'multiagent': {
            # list of all policies
            "policies": {
                "pol": (None, obs_space, act_space, {}),
            },
            # the mapping function between agents and policies. We map all agents to 1 policy.
            "policy_mapping_fn": lambda x: "pol",
        }
    }

    # call tune.run() to run the coop_train_fn() with the config() above
    tune.run(testspeed_ppo_coop_train, config=config)
