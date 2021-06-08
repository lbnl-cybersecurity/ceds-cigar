from pycigar.utils.input_parser import input_parser

di = '/home/toanngo/Documents/GitHub/ceds-cigar/pycigar'
misc_inputs = di +'/data/ieee37busdata/misc_inputs.csv'
dss = di + '/data/ieee37busdata/ieee37.dss'
load_solar = di + '/data/ieee37busdata/load_solar_data.csv'
breakpoints = di + '/data/ieee37busdata/breakpoints.csv'

start = 100
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True)
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
del sim_params['attack_randomization']

from pycigar.envs import CentralControlPVInverterEnv

env = CentralControlPVInverterEnv(sim_params=sim_params)

env.reset()
done = False
while not done:
    _, _, done, _ = env.step(10)