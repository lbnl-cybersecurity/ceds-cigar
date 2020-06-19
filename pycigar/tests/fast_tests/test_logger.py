import unittest

import numpy as np
from pycigar.utils.logging import logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = logger()
        self.logger.set_active()
        self.logger.reset()

    def test_log(self):
        values = [0.1, -0.1, None, 0, np.Inf]
        for v in values:
            self.logger.log('object1', 'param1', v)
            self.logger.log('object1', 'param2', v)
            self.logger.log('object2', 'param1', v)

        self.assertListEqual(self.logger.log_dict['object1']['param1'], values)
        self.assertListEqual(self.logger.log_dict['object1']['param2'], values)
        self.assertListEqual(self.logger.log_dict['object2']['param1'], values)

    def test_reset(self):
        self.logger.log('object', 'param', 0.1)
        self.logger.custom_metrics['custom'] = 0.1
        self.logger.reset()
        self.assertDictEqual(self.logger.log_dict, {}, 'Log dict should be empty after reset')
        self.assertDictEqual(self.logger.custom_metrics, {}, 'Log custom metrics should be empty after reset')

    def test_active(self):
        values = [0.1, -0.1, None, 0, np.Inf]
        self.logger.set_active(False)
        for v in values:
            self.logger.log('object1', 'param1', v)
            self.logger.log('object1', 'param2', v)
            self.logger.log('object2', 'param1', v)

        self.logger.set_active()
        for v in values:
            self.logger.log('object1', 'param1', v)
            self.logger.log('object1', 'param2', v)
            self.logger.log('object2', 'param1', v)

        self.assertListEqual(self.logger.log_dict['object1']['param1'], values)
        self.assertListEqual(self.logger.log_dict['object1']['param2'], values)
        self.assertListEqual(self.logger.log_dict['object2']['param1'], values)

    def test_log_single(self):
        values = [0.1, -0.1, None, 0, np.Inf]
        for v in values:
            self.logger.log_single('object1', 'param1', v)
            self.logger.log_single('object1', 'param2', v)
            self.logger.log_single('object2', 'param1', v)

        self.assertEqual(self.logger.log_dict['object1']['param1'], values[-1])
        self.assertEqual(self.logger.log_dict['object1']['param2'], values[-1])
        self.assertEqual(self.logger.log_dict['object2']['param1'], values[-1])


if __name__ == "__main__":
    unittest.main()
