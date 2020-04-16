import traceback
import numpy as np
import atexit
import gym
from gym.spaces import Box
from pycigar.core.kernel.kernel import Kernel
from pycigar.utils.logging import logger


class Env(gym.Env):

    """Base environment for PyCIGAR, only have 1 agent.

    Attributes
    ----------
    env_time : int
        Environment time, it may be different from the simulation time.
        We can run a few timesteps in the simulation to warm up, but the environment time only increase when we call
        step.
    k : Kernel
        PyCIGAR kernel, abstract function calling across grid simulator APIs.
    sim_params : dict
        A dictionary of simulation information. for example: /examples/rl_config_scenarios.yaml
    simulator : str
        The name of simulator we want to use, by default it is OpenDSS.
    state : TYPE
        The state of environment after performing one step forward.
    """

    def __init__(self, sim_params, simulator='opendss'):
        """Initialize the environment.

        Parameters
        ----------
        sim_params : dict
            A dictionary of simulation information. for example: /examples/rl_config_scenarios.yaml
        simulator : str
            The name of simulator we want to use, by default it is OpenDSS.
        """
        self.state = None
        self.simulator = simulator

        # initialize the kernel
        self.k = Kernel(simulator=self.simulator,
                        sim_params=sim_params)

        # start an instance of the simulator (ex. OpenDSS)
        kernel_api = self.k.simulation.start_simulation()
        # pass the API to all sub-kernels
        self.k.pass_api(kernel_api)
        # start the corresponding scenario
        # self.k.scenario.start_scenario()

        # when exit the environment, trigger function terminate to clear all attached processes.
        atexit.register(self.terminate)

        self.post_process_output_specs = False

    def restart_simulation(self, sim_params):
        """Not in use.
        """
        pass

    def setup_initial_state(self):
        """Not in use.
        """
        pass

    def step(self, rl_actions):
        raise NotImplementedError

    def reset(self):
        self.env_time = 0
        self.sim_params = self.k.update(reset=True)  # hotfix: return new sim_params sample in kernel?
        states = self.get_state()

        self.INIT_ACTION = {}
        pv_device_ids = self.k.device.get_pv_device_ids()
        for device_id in pv_device_ids:
            self.INIT_ACTION[device_id] = np.array(self.k.device.get_control_setting(device_id))
        return states

    def additional_command(self):
        pass

    def clip_actions(self, rl_actions=None):
        if rl_actions is None:
            return None

        if isinstance(self.action_space, Box):
            if type(rl_actions) is dict:
                for key, action in rl_actions.items():
                    rl_actions[key] = np.clip(
                        action,
                        a_min=self.action_space.low,
                        a_max=self.action_space.high)
            else:
                rl_actions = np.clip(
                    rl_actions,
                    a_min=self.action_space.low,
                    a_max=self.action_space.high)
        return rl_actions

    def apply_rl_actions(self, rl_actions=None):
        if rl_actions is None:
            return None

        rl_clipped = self.clip_actions(rl_actions)
        self._apply_rl_actions(rl_clipped)

    def _apply_rl_actions(self, rl_actions):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        return NotImplementedError

    def compute_reward(self, rl_actions, **kwargs):
        return 0

    def terminate(self):
        try:
            # close everything within the kernel
            self.k.close()
        except FileNotFoundError:
            print(traceback.format_exc())

    @property
    def base_env(self):
        return self
