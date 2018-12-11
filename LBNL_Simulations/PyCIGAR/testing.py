
from utils.misc.parser import parser
import numpy as np
from utils.simulation.simulation import simulation

args = parser('config_file.csv')
print(args)

sim = simulation(**args, verbose=False, addNoise=False)

for ep in range(10):
    end = False
    while not end:
        end = sim.runOneStep()
    sim.reset()
    print(ep)
 
