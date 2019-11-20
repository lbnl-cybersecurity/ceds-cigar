from pycigar.utils.registry import make_create_env
import yaml
from ray.tune.registry import register_env
from ray import tune
from ray.tune import Trainable


class BenchmarkAdaptiveControlPVInverter(Trainable):

    def _setup(self, config):
        self.create_env, env_name = make_create_env(params=config, version=0)
        register_env(env_name, self.create_env)
        self.total_reward = 0.0  # end = 1000

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


stream = open("benchmark_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

pycigar_params = {"env_name": "AdaptiveControlPVInverterEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ["pv_8", "pv_9"]}

pycigar_params['sim_params']['scenario_config']['custom_configs']['load_scaling_factor'] = tune.grid_search([1, 2, 3])
pycigar_params['sim_params']['scenario_config']['custom_configs']['solar_scaling_factor'] = tune.grid_search([1, 2, 3])

analysis = tune.run(BenchmarkAdaptiveControlPVInverter,
                    config=pycigar_params)
