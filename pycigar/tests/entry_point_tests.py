import unittest

import pycigar

loader = unittest.TestLoader()
start_dir = pycigar.PROJECT_DIR + '/tests/fast_tests'
print(start_dir)
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)

# Test no-agent path
pycigar.main(
    pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv",
    pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss",
    pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv",
    pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv",
    2,
    None,
    pycigar.LOG_DIR,
)

# Test the training
# pycigar.main(
#    pycigar.pycigdir + "/data/ieee37busdata/misc_inputs.csv",
#    pycigar.pycigdir + "/data/ieee37busdata/ieee37.dss",
#    pycigar.pycigdir + "/data/ieee37busdata/load_solar_data.csv",
#    pycigar.pycigdir + "/data/ieee37busdata/breakpoints.csv",
#    0,
#    pycigar.pycigdir + "/result/policy/",
#    pycigar.pycigdir + "/result/",  # output dir
# )

# Test running with a trained agent
# pycigar.main(
#     pycigar.DATA_DIR + "/ieee37busdata/misc_inputs.csv",
#     pycigar.DATA_DIR + "/ieee37busdata/ieee37.dss",
#     pycigar.DATA_DIR + "/ieee37busdata/load_solar_data.csv",
#     pycigar.DATA_DIR + "/ieee37busdata/breakpoints.csv",
#     1,
#     pycigar.PROJECT_DIR + "/docs/SAMPLE_RESULT_policy/",
#     pycigar.LOG_DIR,  # output dir
# )

# TODO: clean up the API to handle optional arguments.
