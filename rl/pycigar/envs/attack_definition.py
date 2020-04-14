"""Contains different attack definition generator classes to introduce more variability into attack scenarios"""
import random
from typing import List, Union


class AttackDefinitionGenerator:
    """Generates new attack definitions for a simulation scenario

    Attributes:
    ---------
    start_time: int
        start time of the simulation scenario
    end_time: int
        end time of the simulation scenario

    """

    def __init__(
            self,
            start_time: int,
            end_time: int):
        self.start_time = start_time
        self.end_time = end_time

    def new_dev_hack_info(self) -> List[Union[int, float]]:
        """
        Randomly generates hacking info for one adversary inverter

        Returns:
        -------
        list
            a list ("hack_start": int, "percentage": float in (0, 1), "hack_end": int)
        """
        duration = self.end_time - self.start_time
        # relative to self.start_time, anything from 0 to end - start
        hack_start = random.randint(250, 250 + 10)  # random.randint(int(duration*2/5), int(duration*2/5)+10)
        # hack_duration = random.randint(int(duration*4/5), )
        # hack end is relative to start time, can be greater than duration,
        # in this we assume that it's hacked during the whole length of the episode
        # hack_end = hack_start + hack_duration
        hack_end = random.randint(500, 500 + 10)  # random.randint(int(duration*4/5), int(duration*4/5)+10)
        # percentage can be 0.1, 0.2, ..., 0.9
        percentage = random.randint(4, 5) / 10
        res = [
            hack_start,
            percentage,
            hack_end
        ]

        return res


if __name__ == "__main__":
    attack_def = AttackDefinitionGenerator(0, 1440)
    print(attack_def.new_dev_hack_info())
    print(attack_def.new_dev_hack_info())
    print(attack_def.new_dev_hack_info())
