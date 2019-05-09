# from utils.importConstants import *

import numpy as np
import matplotlib.pyplot as plt
from math import tan, acos
import copy
import pandas as pd
import time
import opendssdirect as dss
from opendssdirect.utils import Iterator

dss.run_command('Redirect feeder/LVTestCaseNorthAmerican/Master.dss')
# dss.Text.Command('BatchEdit RegControl..* enabled= No') # Disabling all regulator controls
# dss.Text.Command('BatchEdit CapControl..* enabled= No') # Disabling all capacitor controls
dss.Solution.Solve()

if not dss.Solution.Converged():
    print('Initial Solution Not Converged. Check Model for Convergence')
else:
    print('Initial Model Converged. Proceeding to Next Step.')
    TotalLoads = dss.Loads.Count()
    AllLoadNames = dss.Loads.AllNames()
    print('OpenDSS Model Compilation Done.')

load_kW = [i() for i in Iterator(dss.Loads, 'kW')]
print(load_kW)
#
# dss.Circuit.SetActiveElement('load.S65a')
# dss.Circuit.SetActiveBus(dss.CktElement.BusNames()[0])
# voltage = dss.Bus.puVmagAngle()[::2]
# voltage = np.mean(voltage)
# print(voltage)
print(len(dss.Circuit.AllBusNames()))
