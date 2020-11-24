from pycigar.utils.input_parser import input_parser
from pycigar.utils.logging import logger
import time


misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/misc_inputs.csv'
dss = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37.dss'
#dss_b = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37_b.dss'
#dss_c = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37_c.dss'
load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/load_solar_data.csv'
breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/breakpoints.csv'


#misc_inputs = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee240busdata/misc_inputs.csv'
#dss = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee240busdata/Master.dss'
#dss_b = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37_b.dss'
#dss_c = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee37busdata_regulator_attack/ieee37_c.dss'
#load_solar = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee240busdata/load_solar_data.csv'
#breakpoints = '/home/toanngo/Documents/GitHub/cigar-document/ceds-cigar/pycigar/data/ieee240busdata/breakpoints.csv'


start = 100
sim_params = input_parser(misc_inputs, dss, load_solar, breakpoints, benchmark=True, vectorized_mode=True, percentage_hack=0.45)
#sim_params['simulation_config']['network_model_directory'] = [dss, dss_b, dss_c]
sim_params['scenario_config']['start_end_time'] = [start, start + 750]
sim_params['simulation_config']['custom_configs']['solution_control_mode'] = 2
del sim_params['attack_randomization']

from pycigar.envs import CentralControlPVInverterEnv
from pycigar.envs import CentralControlPhaseSpecificPVInverterEnv



env = CentralControlPVInverterEnv(sim_params=sim_params)
#env = CentralControlPhaseSpecificPVInverterEnv(sim_params=sim_params)

start = time.time()
env.reset()
done = False
while not done:
    _, _, done, _ = env.step(2)

print('total time: ', time.time() - start)
print('step only time: ', logger().custom_metrics['step_only_time'])
#print('pv only time: ', logger().custom_metrics['pv_inverter_time'])
print('opendss only time:', logger().custom_metrics['opendss_time'])
print('opendss solve time:', logger().custom_metrics['opendss_solve_time'])
print('node voltage time:', logger().custom_metrics['node_voltage_time'])



