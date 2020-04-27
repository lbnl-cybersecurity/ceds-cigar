#test on open subprocess to 'opendsscmd', success
#DONE
#import subprocess
#process = subprocess.Popen(['opendsscmd'], shell=True)


#test on TCP port
#status: DONE
#from power.core.kernel.simulation.opendss import OpenDSSSimulation
#from power.core.params import OpenDSSParams
#simulation = OpenDSSSimulation("ahihi")
#params = OpenDSSParams()
#simulation.start_simulation("scenario", params)

#test start OpenDSS and send signal to it
from pycigar.core.kernel.simulation.opendss import OpenDSSSimulation
from pycigar.core.params import OpenDSSParams
simulation = OpenDSSSimulation("ahihi")
params = OpenDSSParams()
simulation.start_simulation("scenario", params)
