from copy import deepcopy
import pycigar.envs
from gym.envs.registration import register
import gym


def make_create_env(pycigar_params, version=0):

    env_name = pycigar_params["env_name"] + '-v{}'.format(version)

    def create_env(config):
        """ the env config is passed to this fn by rllib (env_config key) """

        single_agent_envs = [env for env in dir(pycigar.envs)
                             if not env.startswith('__')]

        assert isinstance(pycigar_params["env_name"], str)
        if pycigar_params['env_name'] in single_agent_envs:
            env_loc = 'pycigar.envs'
        else:
            env_loc = 'pycigar.envs.multiagent'

        try:
            register(
                id=env_name,
                entry_point=env_loc + ':{}'.format(pycigar_params["env_name"]),
                kwargs={
                    "simulator": pycigar_params['simulator'],
                    "tracking_ids": pycigar_params['tracking_ids']
                })
        except Exception:
            pass
        return gym.envs.make(env_name, sim_params=config)

    return create_env, env_name
