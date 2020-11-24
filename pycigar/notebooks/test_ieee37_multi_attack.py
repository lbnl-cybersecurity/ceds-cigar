from pycigar.utils.input_parser import input_parser

misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
dss = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'

start = 100
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=0.45)
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
del sim_params['attack_randomization']

    
#sim_params['scenario_config']['custom_configs']['slack_bus_voltage'] = 1.04
#sim_params['scenario_config']['custom_configs']['slack_bus_voltage'] = 1.04
#base_config['env_config']['M'] = 15
#base_config['env_config']['N'] = 0.1
#base_config['env_config']['P'] = 18
#base_config['env_config']['Q'] = 100
#base_config['env_config']['T'] = 1500
sim_params['M'] = 15
sim_params['N'] = 0 #0.1
sim_params['P'] = 0 #3
sim_params['Q'] = 0 #25
sim_params['T'] = 100

from pycigar.envs import MultiAttackCentralControlPhaseSpecificPVInverterEnv

env = MultiAttackCentralControlPhaseSpecificPVInverterEnv(sim_params=sim_params)
env.reset()
done = False
total_r = 0
while not done:
    _, r, done, _ = env.step([10, 10, 10])
    total_r += r

print("Attack_type: {}, M: {}, N: {}, P: {}, Q: {}, T: {}, total_r: {}".format(env.k.scenario.choose_attack,
                                                              sim_params['M'],
                                                              sim_params['N'],
                                                              sim_params['P'],
                                                              sim_params['Q'],
                                                              sim_params['T'],
                                                              total_r))