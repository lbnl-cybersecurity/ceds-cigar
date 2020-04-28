from pycigar.utils.registry import make_create_env
import yaml
from ray.tune.registry import register_env
from ray import tune
from ray.tune import Trainable
import numpy as np
import pycigar.config as config
import os
import json


class BenchmarkAdaptiveControlPVInverter(Trainable):

    def _setup(self, config):
        self.create_env, env_name = make_create_env(params=config, version=0)
        register_env(env_name, self.create_env)
        self.total_reward = 0.0  # end = 1000
        self.config = config

    def _train(self):
        env = self.create_env()
        env.reset()
        done = False

        while not done:
            _, reward, done, _ = env.step(None)
            self.total_reward += sum(reward.values())

        return {
            "mean_accuracy": self.total_reward,
            "done": done,
        }

    def _save(self, checkpoint_dir):
        return {
            "accuracy": self.total_reward,
        }

    def _restore(self, checkpoint):
        self.accuracy = checkpoint["accuracy"]

    def _log_result(self, result):
        if self.total_reward < -1000:
            filename = os.path.join(config.LOG_DIR, 'data_{}.yml'.format(self._experiment_id))
            outfile = open(filename, 'w')
            json.dump(self.config['sim_params'], outfile, indent=4)  # default_flow_style=False, sort_keys=False)
            outfile.close()
            # print(self.config['sim_params'])
        self._result_logger.on_result(result)


stream = open("oscillation_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

pycigar_params = {"env_name": "AdaptiveControlPVInverterEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ['pv_8', 'pv_9']}


def setting(x):
    return [x-0.001, x, x, x+0.001, x+0.002]


def slack(x):
    return x


sl = [slack(x) for x in np.linspace(0.95, 1.05, num=25)]
control_setting = [setting(x) for x in np.linspace(0.89, 1.04, num=25)]
pycigar_params['sim_params']['scenario_config']['custom_configs']['slack_bus_voltage'] = tune.grid_search(sl)
pycigar_params['sim_params']['hack_setting']['default_control_setting'] = tune.grid_search(control_setting)

analysis = tune.run(BenchmarkAdaptiveControlPVInverter, config=pycigar_params)
