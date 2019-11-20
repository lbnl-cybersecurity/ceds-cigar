from ray.tune import Trainable


class IACExample(Trainable):
    def _setup(self, config):
        pass

    def _train(self):
        result_dict = {"accuracy": 0.5, "f1": 0.1}
        return result_dict
