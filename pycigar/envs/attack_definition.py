"""Contains different attack definition generator classes to introduce more variability into attack scenarios"""
import random
from typing import List, Union
import numpy as np

class AttackDefinitionGenerator:
    """Generates new attack definitions for a simulation scenario

    Attributes:
    ---------
    start_time: int
        start time of the simulation scenario
    end_time: int
        end time of the simulation scenario

    """

    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    def new_dev_hack_info(self):
        hack_start = random.randint(250, 250 + 10)  # random.randint(int(duration*2/5), int(duration*2/5)+10)
        hack_end = random.randint(500, 500 + 10)  # random.randint(int(duration*4/5), int(duration*4/5)+10)

        percentage = random.randint(40, 50) / 100
        res = [hack_start, percentage, hack_end]

        return res

class AttackDefinitionGeneratorEvaluation:
    """Generates new attack definitions for a simulation scenario

    Attributes:
    ---------
    start_time: int
        start time of the simulation scenario
    end_time: int
        end time of the simulation scenario

    """

    def __init__(self, start_time, end_time):
        self.mode = 0

        duration = end_time - start_time
        percentage = np.linspace(5, 50, 5)/100
        start_time = np.linspace(100, 11000, 5)
        self.scenarios = []
        for p in percentage:
            for s in start_time:
                s = int(s)
                scenarios = [p, s, s + duration]
                self.scenarios.append(scenarios)

    def change_mode(self):
        res = self.scenarios[self.mode][1:]
        self.mode += 1
        if self.mode == len(self.scenarios) - 1:
            self.mode = 0

        return res

    def new_dev_hack_info(self):
        hack_start = random.randint(250, 250 + 10)  # random.randint(int(duration*2/5), int(duration*2/5)+10)
        hack_end = random.randint(500, 500 + 10)  # random.randint(int(duration*4/5), int(duration*4/5)+10)
        percentage = self.scenarios[self.mode][0]
        res = [hack_start, percentage, hack_end]

        return res
if __name__ == "__main__":
    attack_def = AttackDefinitionGenerator(0, 1440)
    print(attack_def.new_dev_hack_info())
    print(attack_def.new_dev_hack_info())
    print(attack_def.new_dev_hack_info())
