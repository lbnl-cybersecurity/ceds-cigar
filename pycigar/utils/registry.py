import gym
from gym.envs.registration import register
from pycigar.utils.pycigar_registration import pycigar_register
import pycigar.envs


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
                    "simulator": pycigar_params['simulator']
                })
        except Exception:
            pass

        return gym.envs.make(env_name, sim_params=config)

    return create_env, env_name


def register_devcon(devcon_name, devcon_class, **kwargs):

    assert isinstance(devcon_name, str)

    try:
        pycigar_register(
            id=devcon_name,
            entry_point='{}:{}'.format(devcon_class.__module__, devcon_class.__name__),
            kwargs=kwargs
            )
    except Exception:
        pass

    return devcon_name
