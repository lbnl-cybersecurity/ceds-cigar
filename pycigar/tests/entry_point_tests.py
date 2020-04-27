import pycigar

# Test no-agent path
pycigar.main(
    pycigar.pycigdir + "/data/ieee37busdata/misc_inputs.csv",
    pycigar.pycigdir + "/data/ieee37busdata/ieee37.dss",
    pycigar.pycigdir + "/data/ieee37busdata/load_solar_data.csv",
    pycigar.pycigdir + "/data/ieee37busdata/breakpoints.csv",
    2,
    None,
    pycigar.pycigdir + "/result/",
)

# Test the training
#pycigar.main(
#    pycigar.pycigdir + "/data/ieee37busdata/misc_inputs.csv",
#    pycigar.pycigdir + "/data/ieee37busdata/ieee37.dss",
#    pycigar.pycigdir + "/data/ieee37busdata/load_solar_data.csv",
#    pycigar.pycigdir + "/data/ieee37busdata/breakpoints.csv",
#    0,
#    pycigar.pycigdir + "/result/policy/",
#    pycigar.pycigdir + "/result/",  # output dir
#)

# Test running with a trained agent
#pycigar.main(
#    pycigar.pycigdir + "/data/ieee37busdata/misc_inputs.csv",
#    pycigar.pycigdir + "/data/ieee37busdata/ieee37.dss",
#    pycigar.pycigdir + "/data/ieee37busdata/load_solar_data.csv",
#    pycigar.pycigdir + "/data/ieee37busdata/breakpoints.csv",
#    1,
#    pycigar.pycigdir + "/docs/SAMPLE_RESULT_policy/",
#    pycigar.pycigdir + "/result/",  # output dir
#)

# TODO: clean up the API to handle optional arguments.