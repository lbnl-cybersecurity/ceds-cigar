import numpy as np

"""
In the original multiagent environment (the detail implementation is in pycigar/envs/multiagent/base.py and pycigar/envs/base.py),
the details for observation , action and reward are as following:

    - Observation: a dictionary of local observation for each agent, in the form of {'id_1': obs1, 'id_2': obs2,...}.
                   each agent observation is an array of [voltage, solar_generation, y, p_inject, q_inject]
                   at current timestep,
      Example:
        >>> {'pv_5': array([ 1.02470216e+00,  6.73415767e+01,  2.37637525e-02, -6.70097474e+01, 2.28703002e+01]),
             'pv_6': array([ 1.02461386e+00,  7.42201160e+01,  2.36835291e-02, -7.42163505e+01, 2.35234973e+01]), ...}

      To have a valid openAI gym environment, we have to declare the observation space, what value and dimension will be valid as
      a right observation.

      The observation space is:
        Box(low=-float('inf'), high=float('inf'), shape=(5, ), dtype=np.float64),
      which describe an array of 5, each value can range from -infinity to infinity.

    - Reward: a dictionary of reward for each agent, in the form of {'id_1': reward1, 'id_2': reward2,...}
              default reward for each agent is 0. Depending on our need, we will write suitable wrappers.
      Example:
        >>> {'pv_5': 0, 'pv_6': 0,...}

    - Action: a dictionary of action for each agent, in for form of {'id_1': act1, 'id_2': act2,...}
              there are 2 forms of action: the actions return from RLlib and the actions we use to feed into our environment.

              the actions return from RLlib can be anything (only one value of controlled breakpoint,
              or 5-discretized values of controlled breakpoints) but before we feed the actions to the environment,
              we transform it into the valid form which the environment can execute. For inverter, the control is
              an array of 5 breakpoints which is the actual setting of inverter that we want.
              A valid form of action to feed to the environment: {'id_1': arrayof5, 'id_2': arrayof5,...}
      Example:
        >>> {'pv_5': anp.array([0.98, 1.01, 1.01, 1.04, 1.08]),
             'pv_6': np.array([0.90, 1.00, 1.02, 1.03, 1.07]), ...}

Base on the original multiagent environment observation, action and reward,
the wrappers are used to change the observation, action and reward as we need for different experiment.
"""


class Wrapper:
    """
    The abstract definition of wrapper.
    """

    def __init__(self, env):
        self.env = env

    def step(self, rl_actions, randomize_rl_update=None):
        return self.env.step(rl_actions, randomize_rl_update)

    def reset(self):
        observation = self.env.reset()
        return observation

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)
