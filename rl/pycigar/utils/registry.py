from copy import deepcopy
import pycigar.envs
from gym.envs.registration import register
import gym


def make_create_env(params, version=0, render=None):

    env_name = params["env_name"] + '-v{}'.format(version)

    def create_env(*_):
        sim_params = deepcopy(params['sim_params'])
        single_agent_envs = [env for env in dir(pycigar.envs)
                             if not env.startswith('__')]

        if isinstance(params["env_name"], str):
            if params['env_name'] in single_agent_envs:
                env_loc = 'pycigar.envs'
            else:
                env_loc = 'pycigar.envs.multiagent'

        try:
            register(
                id=env_name,
                entry_point=env_loc + ':{}'.format(params["env_name"]),
                kwargs={
                    "sim_params": sim_params,
                    "simulator": params['simulator'],
                    "tracking_ids": params['tracking_ids']
                })
        except Exception:
            pass
        return gym.envs.make(env_name)

    return create_env, env_name
