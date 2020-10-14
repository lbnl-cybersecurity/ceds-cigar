import unittest

import pycigar
from pycigar.utils.input_parser import input_parser
from pycigar.utils.logging import logger
from pycigar.utils.registry import make_create_env


class TestLogging(unittest.TestCase):
    def setUp(self):
        pycigar_params = {
            'exp_tag': 'cooperative_multiagent_ppo',
            'env_name': 'CentralControlPVInverterEnv',
            'simulator': 'opendss',
        }

        create_env, env_name = make_create_env(pycigar_params, version=0)

        misc_inputs_path = pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv"
        dss_path = pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss"
        load_solar_path = pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv"
        breakpoints_path = pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv"

        sim_params = input_parser(misc_inputs_path, dss_path, load_solar_path, breakpoints_path)
        self.env = create_env(sim_params)

    def test_nodes(self):
        obs = self.env.reset()
        obs, r, done, _ = self.env.step(self.env.init_action)

        log = logger().log_dict
        for k in log:
            if 'node' in log[k]:
                self.assertTrue(log[k]['node'] in log, msg='Nodes should be logged')

    def test_lengths(self):
        obs = self.env.reset()
        obs, r, done, _ = self.env.step(self.env.init_action)
        log = logger().log_dict
        for k in log:
            if 'node' in log[k]:  # device
                lengths = [
                    len(log[k][s])
                    for s in ['y', 'u', 'p_set', 'q_set', 'p_out', 'q_out', 'control_setting', 'solar_irr']
                ]
                self.assertTrue(
                    max(lengths) == min(lengths), msg='There may be a problem with device logging frequency'
                )
            elif 'kw' in log[k]:  # node
                lengths = [len(log[k][s]) for s in ['p', 'q', 'kw', 'kvar', 'voltage']]
                self.assertTrue(max(lengths) == min(lengths), msg='There may be a problem with node logging frequency')

        if 'network' in log:
            lengths = [
                len(log['network'][s])
                for s in ['substation_power', 'loss', 'substation_top_voltage', 'substation_bottom_voltage']
            ]
            self.assertTrue(max(lengths) == min(lengths), msg='There may be a problem with network logging frequency')


if __name__ == "__main__":
    unittest.main()
