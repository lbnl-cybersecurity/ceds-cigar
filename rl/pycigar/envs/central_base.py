import traceback
import numpy as np
import atexit
import gym
from gym.spaces import Box
from pycigar.core.kernel.kernel import Kernel
import matplotlib.pyplot as plt
from datetime import datetime
import pycigar.config as config
import os
from scipy import signal
import random
ACTION_RANGE = 0.1
ACTION_STEP = 0.05

class CentralEnv(gym.Env):

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
    tracking_ids : list
        A list of agent ids we want to keep track.
    tracking_infos : dict
        The tracking information of the agent ids.
    """

    def __init__(self, sim_params, simulator='opendss', tracking_ids=None):
        """Initialize the environment.

        Parameters
        ----------
        sim_params : dict
            A dictionary of simulation information. for example: /examples/rl_config_scenarios.yaml
        simulator : str
            The name of simulator we want to use, by default it is OpenDSS.
        tracking_ids : list
            A list of agent ids we want to keep track.
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
        #self.k.scenario.start_scenario()

        #INIT_ACTION template

        # when exit the environment, trigger function terminate to clear all attached processes.
        atexit.register(self.terminate)

        # save the tracking ids, we will keep track the history of agents who have the ids in this list.
        self.tracking_ids = tracking_ids

    def restart_simulation(self, sim_params, render=None):
        """Not in use.
        """
        pass

    def setup_initial_state(self):
        """Not in use.
        """
        pass

    def step(self, rl_actions):
        """Move the environment one step forward.

        Parameters
        ----------
        rl_actions : dict
            A dictionary of actions of each agents controlled by RL algorithms.

        Returns
        -------
        Tuple
            A tuple of (obs, reward, done, infos).
            obs: a dictionary of new observation from the environment.
            reward: a dictionary of reward received by agents.
            done: bool
        """
        #print(self.k.time)
        rl_actions = self.action_mapping(rl_actions)

        next_observation = None
        self.old_actions = {}
        for rl_id in self.k.device.get_rl_device_ids():
            self.old_actions[rl_id] = self.k.device.get_control_setting(rl_id)
        randomize_rl_update = {}
        if rl_actions is not None:
            for rl_id in self.k.device.get_rl_device_ids():
                randomize_rl_update[rl_id] = np.random.randint(low=0, high=3)
        else:
            rl_actions = self.old_actions

        count = 0 

        for _ in range(self.sim_params['env_config']["sims_per_step"]):
            if self.tracking_ids is not None:
                self.pycigar_tracking()
            self.env_time += 1

            # perform action update for PV inverter device
            if len(self.k.device.get_adaptive_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_adaptive_device_ids():
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_adaptive_device_ids(), control_setting)

            # perform action update for PV inverter device
            if len(self.k.device.get_fixed_device_ids()) > 0:
                control_setting = []
                for device_id in self.k.device.get_fixed_device_ids():
                    action = self.k.device.get_controller(device_id).get_action(self)
                    control_setting.append(action)
                self.k.device.apply_control(self.k.device.get_fixed_device_ids(), control_setting)

            temp_rl_actions = {}
            for rl_id in self.k.device.get_rl_device_ids():
                temp_rl_actions[rl_id] = rl_actions[rl_id]  

            # perform action update for PV inverter device controlled by RL control
            rl_dict = {}
            for rl_id in temp_rl_actions.keys():
                if randomize_rl_update[rl_id] == 0:
                    rl_dict[rl_id] = temp_rl_actions[rl_id]
                else:
                    randomize_rl_update[rl_id] -=1

            for rl_id in rl_dict.keys():
                del temp_rl_actions[rl_id]
                
            self.apply_rl_actions(rl_dict)
            self.additional_command()
            
            if self.k.time <= self.k.t:
                self.k.update(reset=False)

                # check whether the simulator sucessfully solved the powerflow
                converged = self.k.simulation.check_converged()
                if not converged:
                    break

                states = self.get_state()
                if next_observation is None:
                    next_observation = states
                else:
                    next_observation += states
                count += 1

            if self.k.time >= self.k.t:
                break


        next_observation = next_observation/count 
                
        # the episode will be finished if it is not converged.
        finish = not converged or (self.k.time == self.k.t)

        if finish:
            done = True
        else:
            done = False

        # we push the old action, current action, voltage, y, p_inject, p_max into additional info, which will return by the env after calling step.
        #infos = {key: {'voltage': self.k.node.get_node_voltage(self.k.device.get_node_connected_to(key)),
        #               'y': self.k.device.get_device_y(key),
        #               'p_inject': self.k.device.get_device_p_injection(key),
        #               'p_max': self.k.device.get_solar_generation(key),
        #               'env_time': self.env_time,
        #               } for key in states.keys()}
        infos = {key: {'voltage': self.k.node.get_node_voltage(self.k.device.get_node_connected_to(key)),
                       'y': self.k.device.get_device_y(key),
                       'p_inject': self.k.device.get_device_p_injection(key),
                       'p_max': self.k.device.get_device_q_injection(key),
                       'env_time': self.env_time,
                       } for key in self.k.device.get_rl_device_ids()}

        for key in self.k.device.get_rl_device_ids():
            if self.old_actions is not None:
                infos[key]['old_action'] = self.old_actions[key]
            else:
                infos[key]['old_action'] = None
            if rl_actions is not None:
                infos[key]['current_action'] = rl_actions[key]
            else:
                infos[key]['current_action'] = None

        # clip the action into a good range or not
        if self.sim_params['env_config']['clip_actions']:
            rl_clipped = self.clip_actions(rl_actions)
            reward = self.compute_reward(rl_clipped, fail=not converged)
        else:
            reward = self.compute_reward(rl_actions, fail=not converged)
        
        # tracking
        if self.tracking_ids is not None:
            self.pycigar_tracking()
        
        return next_observation, reward, done, infos

    def reset(self):
        self.env_time = 0
        self.sim_params = self.k.update(reset=True)  # hotfix: return new sim_params sample in kernel?
        states = self.get_state()
        
        # tracking
        if self.tracking_ids is not None:
            self.pycigar_tracking()

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

    def get_kernel(self):
        return self.k

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

    def pycigar_tracking(self):
        """Keep track the change on list of agent ids.
        The tracking value is voltage, y-value, p inject and action for the whole simulation.
        """
        if self.env_time == 0:
            self.tracking_infos = {}
            for tracking_id in self.tracking_ids:
                self.tracking_infos[tracking_id] = dict(v_val=[],
                                                        y_val=[],
                                                        q_set=[],
                                                        q_val=[],
                                                        a_val=[])

        for tracking_id in self.tracking_ids:
            node_id = self.k.device.get_node_connected_to(tracking_id)
            self.tracking_infos[tracking_id]['v_val'].append(self.k.node.get_node_voltage(node_id))
            self.tracking_infos[tracking_id]['y_val'].append(self.k.device.get_device_y(tracking_id))
            #p_max = self.k.device.get_solar_generation(tracking_id)
            #p_inject = self.k.device.get_device_p_injection(tracking_id)
            self.tracking_infos[tracking_id]['q_set'].append(self.k.device.get_device_q_set(tracking_id))
            self.tracking_infos[tracking_id]['q_val'].append(self.k.device.get_device_q_injection(tracking_id))
            self.tracking_infos[tracking_id]['a_val'].append(list(self.k.device.get_control_setting(tracking_id)))

    def plot(self, exp_tag='', env_name='', iteration=0, reward=0):
        """Plot the result of tracking ids after the simulation.

        Parameters
        ----------
        exp_tag : str, optional
            The experiment tag, this will be used as a folder name created under /result/.
        env_name : str, optional
            Name of the environment which we run the simulation.
        iteration : int, optional
            The number of training iteration taken place before this plot.
        """
        num_col = len(self.tracking_infos.keys())
        if num_col != 1:
            f, ax = plt.subplots(3, num_col, figsize=(25, 20))
            for col in range(num_col):
                tracking_id = list(self.tracking_infos.keys())[col]
                ax[0, col].set_title(tracking_id + " -- total reward: " + str(reward))
                ax[0, col].plot(self.tracking_infos[tracking_id]['v_val'])
                ax[0, col].set_ylabel('voltage')
                ax[1, col].plot(self.tracking_infos[tracking_id]['y_val'])
                ax[1, col].set_ylabel('oscillation observer')
                #ax[2, col].plot(self.tracking_infos[tracking_id]['p_val'])
                #ax[2, col].set_ylabel('(1 + p_inject/p_max)**2')
                labels = ['a1', 'a2', 'a3', 'a4', 'a5']
                [a1, a2, a3, a4, a5] = ax[2, col].plot(self.tracking_infos[tracking_id]['a_val'])
                ax[2, col].set_ylabel('action')
                plt.legend([a1, a2, a3, a4, a5], labels, loc=1)
        else:
            f, ax = plt.subplots(4, figsize=(25, 25))
            for col in range(num_col):
                tracking_id = list(self.tracking_infos.keys())[col]
                ax[0].set_title(tracking_id + " -- total reward: " + str(reward))
                ax[0].plot(self.tracking_infos[tracking_id]['v_val'])
                ax[0].set_ylabel('voltage')
                ax[0].grid(b=True, which='both')
                ax[1].plot(self.tracking_infos[tracking_id]['y_val'])
                ax[1].set_ylabel('oscillation observer')
                ax[1].grid(b=True, which='both')
                ax[2].plot(self.tracking_infos[tracking_id]['q_set'])
                ax[2].plot(self.tracking_infos[tracking_id]['q_val'])
                ax[2].set_ylabel('reactive power')
                ax[2].grid(b=True, which='both')
                labels = ['a1', 'a2', 'a3', 'a4', 'a5']
                [a1, a2, a3, a4, a5] = ax[3].plot(self.tracking_infos[tracking_id]['a_val'])
                ax[3].set_ylabel('action')
                ax[3].grid(b=True, which='both')
                plt.legend([a1, a2, a3, a4, a5], labels, loc=1)

                #np.savetxt(os.path.join(os.path.join(config.LOG_DIR, exp_tag), 'voltage_profile.txt'), self.tracking_infos[tracking_id]['v_val'])

        if not os.path.exists(os.path.join(config.LOG_DIR, exp_tag)):
            os.makedirs(os.path.join(config.LOG_DIR, exp_tag))
        save_path = os.path.join(os.path.join(config.LOG_DIR, exp_tag), '{}_{}_result_{}.png'.format(exp_tag, env_name, iteration))#, datetime.now().strftime("%H:%M:%S.%f_%d-%m-%Y")))

        f.savefig(save_path)
        plt.close(f)

    def action_mapping(self, rl_actions):
        if rl_actions is None:
            return None
        new_rl_actions = {}
        for rl_id in self.INIT_ACTION.keys():
            new_rl_actions[rl_id] =  self.INIT_ACTION[rl_id] - ACTION_RANGE + ACTION_STEP*rl_actions

        return new_rl_actions
