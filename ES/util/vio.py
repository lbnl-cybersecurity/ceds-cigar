import numpy as np
import util.signal_processing as signal_processing

class voltage_imbalance_observer_manager():

    def __init__(self, time, Ts, jsondata):

        self.violist = []

        self.time = time
        self.Ts = Ts

        self.jsondata = jsondata

    def parse_json(self):

        self.viosjson = self.jsondata['vios']

        for k1 in range(len(self.viosjson)):

            tempvio = voltage_imbalance_observer()

            Toff = 1/100*np.floor(10*np.random.rand())

            tempvio.set_timesteps(self.Ts,self.time,len(self.time))
            # tempesc.set_opertime(self.escsjson[k1]['Top'],self.escsjson[k1]['Toff'])
            tempvio.set_opertime(self.viosjson[k1]['Top'],Toff)

            
            tempvio.set_busname(str(self.viosjson[k1]['bus']))
            tempvio.set_conn(self.viosjson[k1]['conn'])

            tempvio.set_phase(np.asarray(self.viosjson[k1]['phase'].split('.'), dtype=int))
            print(np.asarray(self.viosjson[k1]['phase'].split('.'), dtype=int))

            self.violist.append(tempvio)

        return self.violist


class voltage_imbalance_observer():

    def __init__(self):
        
        pass
    
    def set_timesteps(self, Ts, time, numTimeSteps):
        
        self.Ts = Ts
        
        self.numTimeSteps = numTimeSteps
        
        self.Vimb = np.zeros((numTimeSteps,4))

        self.Vimblpf = np.zeros((numTimeSteps,4))
       
        self.kop = 0
        self.timeop = np.zeros(numTimeSteps)
        self.Nop = 0

    def set_opertime(self, Top, Toff):
        
        self.kop = 0
        
        self.Top = Top
        self.Toff = Toff
        
        self.Tlast = self.Toff
        
        self.flpf = 2.0
        self.wlpf = 2*np.pi*self.flpf
        
    # set the index of the node where the inverter is located
    def set_busidx(self, busidx):
        self.busidx = busidx
        
    # set the name of the node where the inverter is located
    def set_busname(self, busname):
        self.busname = busname

    # set the connection of the node where the inverter is located
    def set_conn(self, conn):
        self.conn = conn

    # set the phase of the node where the inverter is located
    def set_phase(self, phase):
        self.phase = phase

    def observe_voltage(self, kop, Vmag):

        Vab = 0
        Vbc = 0
        Vca = 0
        Vabc = 0

        # if self.phase == '1.2' and len(Vmag) == 2:
        #     Vab = ((Vmag[0] - Vmag[1])**2) / 0.01**2
        # elif self.phase == '1.2' and len(Vmag) == 3:
        #     Vab = ((Vmag[0] - Vmag[1])**2) / 0.01**2
        # elif self.phase == '2.3' and len(Vmag) == 2:
        #     Vbc = ((Vmag[0] - Vmag[1])**2) / 0.01**2
        # elif self.phase == '2.3' and len(Vmag) == 3:
        #     Vbc = ((Vmag[1] - Vmag[2])**2) / 0.01**2
        # elif self.phase == '3.1' and len(Vmag) == 2:
        #     Vca = ((Vmag[1] - Vmag[0])**2) / 0.01**2
        # elif self.phase == '3.1' and len(Vmag) == 3:
        #     Vca = ((Vmag[2] - Vmag[0])**2) / 0.01**2
        # elif self.phase == '1.2.3' and len(Vmag) == 3:
        #     Vab = ((Vmag[0] - Vmag[1])**2) / 0.01**2
        #     Vbc = ((Vmag[1] - Vmag[2])**2) / 0.01**2
        #     Vca = ((Vmag[2] - Vmag[0])**2) / 0.01**2
        #     Vabc = Vab + Vbc + Vca

        if np.all(self.phase == [1, 2]) and len(Vmag) == 2:
            Vab = ((Vmag[0] - Vmag[1])**2) / 0.01**2
        elif np.all(self.phase == [1, 2]) and len(Vmag) == 3:
            Vab = ((Vmag[0] - Vmag[1])**2) / 0.01**2
        elif np.all(self.phase == [2, 3]) and len(Vmag) == 2:
            Vbc = ((Vmag[0] - Vmag[1])**2) / 0.01**2
        elif np.all(self.phase == [2, 3]) and len(Vmag) == 3:
            Vbc = ((Vmag[1] - Vmag[2])**2) / 0.01**2
        elif np.all(self.phase == [3, 1]) and len(Vmag) == 2:
            Vca = ((Vmag[1] - Vmag[0])**2) / 0.01**2
        elif np.all(self.phase == [3, 1]) and len(Vmag) == 3:
            Vca = ((Vmag[2] - Vmag[0])**2) / 0.01**2
        elif np.all(self.phase == [1, 2, 3]) and len(Vmag) == 3:
            Vab = ((Vmag[0] - Vmag[1])**2) / 0.01**2
            Vbc = ((Vmag[1] - Vmag[2])**2) / 0.01**2
            Vca = ((Vmag[2] - Vmag[0])**2) / 0.01**2
            Vabc = Vab + Vbc + Vca
        
        self.Vimb[kop, :] = [Vab, Vbc, Vca, Vabc]
    
    def truncate_time_data(self):
        
        self.Nop = self.kop
        self.timeop = self.timeop[0:self.Nop+1]
        self.Vimb = self.Vimb[0:self.Nop+1,:]
        self.Vimblpf = self.Vimblpf[0:self.Nop+1,:]
        