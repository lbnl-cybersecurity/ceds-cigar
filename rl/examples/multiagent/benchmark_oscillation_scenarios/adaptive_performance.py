from ray import tune
from ray.tune.registry import register_env
from pycigar.utils.registry import make_create_env
import yaml
import pycigar.config as config_pycigar
import os


def coop_train_fn():
    all_params = []
    filenames = [x for x in os.listdir(os.path.join(config_pycigar.LOG_DIR, 'bad_scenarios/train')) if x.endswith(".yml")]
    for filename in filenames:
        filename = os.path.join(config_pycigar.LOG_DIR, 'bad_scenarios/train/{}'.format(filename))
        stream = open(filename, "r")
        sim_params = yaml.safe_load(stream)

        pycigar_params = {"env_name": "AdaptiveControlPVInverterEnv",
                          "sim_params": sim_params,
                          "simulator": "opendss",
                          "tracking_ids": ['pv_8', 'pv_9', 'pv_12'],
                          'filename': filename}

        pycigar_params['sim_params']['scenario_config']['nodes'][5]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][6]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][7]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][8]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][9]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][10]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][11]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][12]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][13]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][14]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][15]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][16]['devices'][0]['controller'] = 'adaptive_inverter_controller'
        pycigar_params['sim_params']['scenario_config']['nodes'][17]['devices'][0]['controller'] = 'adaptive_inverter_controller'

        ADAPTIVEGAIN = 20
        pycigar_params['sim_params']['scenario_config']['nodes'][5]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][6]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][7]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][8]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][9]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][10]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][11]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][12]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][13]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][14]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][15]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][16]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN
        pycigar_params['sim_params']['scenario_config']['nodes'][17]['devices'][0]['custom_configs']['adaptive_gain'] = ADAPTIVEGAIN

        all_params.append(pycigar_params)

    return all_params


def benchmark_adaptive_control_pv_inverter(config):
    create_env, env_name = make_create_env(params=config, version=0)
    register_env(env_name, create_env)

    env = create_env()
    env.reset()
    done = False
    total_reward = 0
    while not done:
        _, reward, done, _ = env.step(None)
        total_reward += sum(reward.values())

    tune.track.log(mean_accuracy=total_reward)


all_params = coop_train_fn()
for pycigar_params in all_params:
    print(pycigar_params['filename'])
    tune.run(benchmark_adaptive_control_pv_inverter, config=pycigar_params)
