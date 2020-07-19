from pycigar.utils.registry import make_create_env
import yaml
from ray.tune.registry import register_env
from ray import tune


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


stream = open("benchmark_config_scenarios.yaml", "r")
sim_params = yaml.safe_load(stream)

pycigar_params = {"env_name": "AdaptiveControlPVInverterEnv",
                  "sim_params": sim_params,
                  "simulator": "opendss",
                  "tracking_ids": ["pv_8", "pv_9", "pv_12"]}

analysis = tune.run(benchmark_adaptive_control_pv_inverter, config=pycigar_params)
