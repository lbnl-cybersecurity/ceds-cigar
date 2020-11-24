from pycigar.utils.input_parser import input_parser

misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata/misc_inputs.csv'
dss = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata/ieee37.dss'
load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata/load_solar_data.csv'
breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata/breakpoints.csv'

start = 100
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=0.45)
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
sim_params['cluster'] = {'1': ['s701a', 's701b', 's701c', 's712c', 's713c', 's714a', 's714b', 's718a', 's720c', 's722b', 's722c', 's724b', 's725b'],
                         '2': ['s727c', 's728', 's729a', 's730c', 's731b', 's732c', 's733a'],
                         '3': ['s734c', 's735c', 's736b', 's737a', 's738a', 's740c', 's741c', 's742a', 's742b', 's744a']
                            }

del sim_params['attack_randomization']


sim_params['M'] = 15
sim_params['N'] = 0 #0.1
sim_params['P'] = 0 #3
sim_params['Q'] = 0 #25
sim_params['T'] = 100

from pycigar.envs.multiagent.multi_envs import ClusterMultiEnv

env = ClusterMultiEnv(sim_params=sim_params)
obs = env.reset()
done = False
total_r = 0
while not done:
    obs, r, done, info = env.step({'agent_1':[10, 10, 10], 'agent_2':[10, 10, 10], 'agent_3':[10, 10, 10]})
    total_r += r