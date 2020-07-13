from ray import tune
import ray
from padice import PADICETrainer
ray.init(local_mode=True)
tune.run(PADICETrainer, config={"env": "Pendulum-v0"})