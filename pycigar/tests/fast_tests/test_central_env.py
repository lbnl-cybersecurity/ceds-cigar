import unittest

import pycigar
from pycigar.utils.input_parser import input_parser
from pycigar.utils.registry import make_create_env
from pycigar.utils.logging import logger
import time
import numpy as np


class TestCentralEnv(unittest.TestCase):
    def setUp(self):
        pycigar_params = {'exp_tag': 'cooperative_multiagent_ppo',
                          'env_name': 'CentralControlPVInverterEnv',
                          'simulator': 'opendss'}

        create_env, env_name = make_create_env(pycigar_params, version=0)

        misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
        dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
        load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
        breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

        sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
        self.env = create_env(sim_params)

    def test_voltages(self):
        done = False
        obs = self.env.reset()
        obs, r, done, _ = self.env.step(self.env.init_action)

        log = logger().log_dict
        self.assertTrue(all([k['voltage'][0] < 1.04 for k in log if 'voltage' in k]))
        print(log)

    def test_start_stability(self):
        done = False
        obs = self.env.reset()
        for _ in range(10):
            obs, r, done, _ = self.env.step(self.env.init_action)

        log = logger().log_dict
        for k in log:
            if 'voltage' in k:
                diff = max(k['voltage']) - min(k['voltage'])
                self.assertAlmostEqual(diff, 0, msg='Voltage should be stable at the beginning')

    def test_init_action_present(self):
        self.env.reset()
        self.assertTrue(hasattr(self.env, 'INIT_ACTION') and isinstance(self.env.INIT_ACTION, dict),
                        msg='No INIT_ACTION dict in environment')

    def test_env_time(self):
        self.env.reset()
        self.assertEqual(self.env.env_time, 0)

    def test_sims_per_step(self):
        self.env.reset()
        done = False
        while not done:
            old_t = self.env.env_time
            obs, r, done, _ = self.env.step(self.env.init_action)
            if not done:
                self.assertEqual(self.env.env_time - old_t, self.env.sim_params['env_config']["sims_per_step"])
            else:
                self.assertLessEqual(self.env.env_time - old_t, self.env.sim_params['env_config']["sims_per_step"])

    def test_sim_time(self):
        self.env.reset()
        t = time.time()
        self.env.step(self.env.init_action)
        tt = (time.time() - t) / self.env.sim_params['env_config']["sims_per_step"]
        self.assertLessEqual(tt, 0.1, msg='Simulation time is greater than 100ms for one step')

if __name__ == "__main__":
    unittest.main()
