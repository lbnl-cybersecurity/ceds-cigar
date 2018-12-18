from utils.misc.build_scenario import *
import opendssdirect as DSSObj
from utils.openDSS.setInfo_linux import *
from utils.openDSS.getInfo_linux import *

from utils.device.Inverter import Inverter
from utils.controller.AdaptiveInvController import AdaptiveInvController
from utils.controller.FixedInvController import FixedInvController
from utils.controller.RLController.VPGInvController import VPGInvController
from utils.controller.RLController.PPOInvController import PPOInvController
from utils.controller.RLController.TRPOInvController import TRPOInvController
from utils.controller.RLController.DDPGInvController import DDPGInvController
from utils.controller.RLController.TD3InvController import TD3InvController

import matplotlib.pyplot as plt
import tensorflow as tf
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs

#init env -> init state, init session
class simulation:
    def __init__(self, **kwargs): 
        self.init = True
        power_factor = 0.9
        self.pf_converted = tan(acos(power_factor))
        
        self.verbose = kwargs['verbose']
        self.openDSSkwargs = kwargs['OpenDSSkwargs']
        self.FileDirectory = kwargs['FileDirectoryBase']
        self.OpenDSSDirectory = kwargs['OpenDSSDirectory']
        self.initController = kwargs['initController']
        self.initHackedController = kwargs['initHackedController']
        self.startTime = kwargs['startTime']
        self.endTime = kwargs['endTime']
        self.addNoise = kwargs['addNoise']
        self.logger_kwargs = kwargs['logger_kwargs']

        self.logger_kwargs = setup_logger_kwargs(**self.logger_kwargs)

        self.RawLoad, self.RawGeneration, self.TotalLoads, self.AllLoadNames = ReadScenarioFile(self.FileDirectory, self.OpenDSSDirectory)
        
        
        self.logger = EpochLogger(**self.logger_kwargs) #take care of logger_kwargs
        self.sess = tf.Session()
        #some default params to convert load
        
        self.reset(addNoise=self.addNoise)
        self.sess.run(tf.global_variables_initializer()) #init for tf incase of using RL
    
    def reset(self, addNoise=True):
        self.timeStep = 0
        self.terminal = False

        Load, Generation = ScenarioInterpolation(self.startTime, self.endTime, self.RawLoad, self.RawGeneration)
        self.Load, self.Generation = ScenarioAddNoise(Load, Generation, self.TotalLoads, addNoise)
        self.TotalTimeSteps = self.Load.shape[0]
        
        MaxGenerationPossible = np.max(self.Generation, axis = 0)
        self.sbar = MaxGenerationPossible
        
        self._restart_openDSS()
        self._restart_node()
        self._restart_controller()
        

    def runOneStep(self):
        
        self._run_solve()
        self._run_hack()
        self._run_pqinjection()
                #if done, print out 'done' or plot if verbose
        if self.timeStep == self.TotalTimeSteps-1:
            self.terminal = True
            if self.verbose:
                self.plot_voltage_at_nodes(list(range(self.TotalLoads)))
                listInv = []
                for key in self.inverters:
                    if self.inverters[key] != []:
                        listInv += [key]
                self.plot_y_at_nodes(listInv)
                self.plot_VBP_at_nodes(listInv)
            self.logger.log('\nSimulation Done!\n')
            return self.terminal

        self.timeStep += 1
        
        return self.terminal

    def _run_solve(self):
        if self.timeStep == 0:
            for node in range(len(self.AllLoadNames)):
                nodeName = self.AllLoadNames[node]
                setLoadInfo(self.DSSObj, [nodeName], 'kw', [self.Load[self.timeStep, node]])
                setLoadInfo(self.DSSObj, [nodeName], 'kvar', [self.pf_converted*self.Load[self.timeStep, node]])
        else:
            for node in range(len(self.AllLoadNames)):
                nodeName = self.AllLoadNames[node]
                setLoadInfo(self.DSSObj, [nodeName], 'kw', [self.Load[self.timeStep, node] + self.nodes[node].at['P', self.timeStep-1]])
                setLoadInfo(self.DSSObj, [nodeName], 'kvar', [self.pf_converted*self.Load[self.timeStep, node] + self.nodes[node].at['Q', self.timeStep-1]])

        self.DSSSolution.Solve()
        if not self.DSSSolution.Converged():
            print('Solution Not Converged at Step:', self.timeStep)

        #get the voltage info
        nodeInfo = getLoadInfo(self.DSSObj, [])
        #distribute voltage to node
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            node.at['Voltage', self.timeStep] = nodeInfo[i]['voltagePU']
            
    def _run_hack(self):
        timeList = list(range(self.TotalTimeSteps))
        
        if self.timeStep in self.initHackedController:
            for node in self.initHackedController[self.timeStep]:
                inverter = self.inverters[node][0]
                hackedInv = {}
                hackedInv['info'] = copy.deepcopy(inverter['info'])
                hackedInv['device'] = copy.deepcopy(inverter['device'])
                
                typeController = self.initHackedController[self.timeStep][node]['type']
                if typeController == 'FixedInvController':
                    hackedInv['controller'] = FixedInvController(timeList, 
                                                             VBP = self.initHackedController[self.timeStep][node]['VBP'],
                                                             device = hackedInv['device'])
                
                for k in range(self.timeStep, self.TotalTimeSteps):
                    hackedInv['info'].loc['sbar'][self.timeStep:] = hackedInv['info'].loc['sbar'][self.timeStep:]*self.initHackedController[self.timeStep][node]['percentHacked']
                    hackedInv['info'].loc['Generation'][self.timeStep:] = hackedInv['info'].loc['Generation'][self.timeStep:]*self.initHackedController[self.timeStep][node]['percentHacked']
                    
                    #generation and sbar change on the original inverter
                    inverter['info'].loc['sbar'][self.timeStep:] = inverter['info'].loc['sbar'][self.timeStep:]*(1-self.initHackedController[self.timeStep][node]['percentHacked'])
                    inverter['info'].loc['Generation'][self.timeStep:] = inverter['info'].loc['Generation'][self.timeStep:]*(1-self.initHackedController[self.timeStep][node]['percentHacked'])

                self.inverters[node].append(hackedInv)                    
    
    def _run_pqinjection(self):
        for node in range(len(self.AllLoadNames)):
            #if we have inverters at that node then...
            if self.inverters[node] != []:
                invertersNode = self.inverters[node] #get the list of inverters at that node
                for inverter in invertersNode: #get an inverter at that node            
                    #################################################
                    device = inverter['device']
                    controller = inverter['controller']
                    info = inverter['info']

                    #calcuate P Q injection by inverter
                    p_inv, q_inv = device.step( v=self.nodes[node].at['Voltage', self.timeStep], 
                                                solar_irr=info.at['Generation', self.timeStep],
                                                solar_minval=5, 
                                                Sbar=info.at['sbar', self.timeStep], 
                                                VBP=controller.get_VBP())

                    #add P Q injection to the node
                    self.nodes[node].at['P', self.timeStep] += p_inv
                    self.nodes[node].at['Q', self.timeStep] += q_inv

                    controller.act( V=self.nodes[node].at['Voltage', self.timeStep],\
                                    G=info.at['Generation', self.timeStep],\
                                    L=self.Load[self.timeStep, node])
    
    def _restart_controller(self):
        features = ['Generation', 'sbar']

        if self.init == True:
            self.inverters = {}
            
            for i in range(len(self.AllLoadNames)):
                self.inverters[i] = []
                if i in self.initController:
                    timeList = list(range(self.TotalTimeSteps))
                    df = pd.DataFrame(columns=timeList,index=features)
                    df.loc['Generation'] = self.Generation[:,i]
                    df.loc['sbar'] = self.sbar[i]

                    inv = {}
                    inv['device'] = Inverter(timeList,lpf_meas=1,lpf_output=0.1)

                    typeController = self.initController[i]['type']
                    if typeController == 'AdaptiveInvController':
                        inv['controller'] = AdaptiveInvController(timeList, VBP = self.initController[i]['VBP'], 
                                                                  device=inv['device'], 
                                                                  delayTimer=self.initController[i]['delayTimer'],
                                                                  nk=self.initController[i]['nk'],
                                                                  threshold=self.initController[i]['threshold'])
                    
                    elif typeController == 'FixedInvController':
                        inv['controller'] = FixedInvController(timeList, VBP = self.initController[i]['VBP'], 
                                                                  device=inv['device'])
                    
                    elif typeController == 'VPGInvController':
                        inv['controller'] = VPGInvController(timeList, VBP = np.array([1.01, 1.03, 1.03, 1.05]), 
                                                             delayTimer=self.initController[i]['delayTimer'], 
                                                             device=inv['device'], 
                                                             sess=self.sess, logger=self.logger)

                    elif typeController == 'PPOInvController':
                        inv['controller'] = PPOInvController(timeList, VBP = np.array([1.01, 1.03, 1.03, 1.05]), 
                                                             delayTimer=self.initController[i]['delayTimer'], 
                                                             device=inv['device'], 
                                                             sess=self.sess, logger=self.logger)

                    elif typeController == 'TRPOInvController':
                        inv['controller'] = TRPOInvController(timeList, VBP = np.array([1.01, 1.03, 1.03, 1.05]), 
                                                             delayTimer=self.initController[i]['delayTimer'], 
                                                             device=inv['device'], 
                                                             sess=self.sess, logger=self.logger)

                    elif typeController == 'DDPGInvController':
                        inv['controller'] = DDPGInvController(timeList, VBP = np.array([1.01, 1.03, 1.03, 1.05]), 
                                                             delayTimer=self.initController[i]['delayTimer'], 
                                                             device=inv['device'], 
                                                             sess=self.sess, logger=self.logger)

                    elif typeController == 'TD3InvController':
                        inv['controller'] = TD3InvController(timeList, VBP = np.array([1.01, 1.03, 1.03, 1.05]), 
                                                             delayTimer=self.initController[i]['delayTimer'], 
                                                             device=inv['device'], 
                                                             sess=self.sess, logger=self.logger)
                    inv['info'] = df
                    self.inverters[i].append(inv)      
            self.init = False
        
        else:
            for i in range(self.TotalLoads):    
                inv = self.inverters[i]
                if i in self.initController:
                    #replace scenarios
                    inv.pop() #delete hacked inv from previous simulation
                    inv = inv[0] 
                    timeList = list(range(self.TotalTimeSteps))
                    df = pd.DataFrame(columns=timeList,index=features)
                    df.loc['Generation'] = self.Generation[:,i]
                    df.loc['sbar'] = self.sbar[i] #reset sbar

                    inv['device'].reset() #reset internal state of device
                    inv['info'] = df #reset info table
                    inv['controller'].reset() #reset controller internal state
          
    def _restart_node(self):
        #init node
        self.nodes = {}
        features = ['Voltage', 'Generation', 'P', 'Q']
        Time = list(range(self.startTime,self.endTime))
        TotalTimeSteps = len(Time)
        
        for i in range(len(self.AllLoadNames)):
            df = pd.DataFrame(columns=list(range(TotalTimeSteps)),index=features)
            self.nodes[i] = df
            self.nodes[i].loc['Generation'] = self.Generation[:,i]
            self.nodes[i].loc['P'] = 0
            self.nodes[i].loc['Q'] = 0
            
    def _restart_openDSS(self):
        DSSObj.run_command('Redirect ' + self.OpenDSSDirectory)
        DSSSolution = DSSObj.Solution
        DSSMon = DSSObj.Monitors
        DSSSolution.Solve()
        if not DSSSolution.Converged():
            print('Initial Solution Not Converged. Check Model for Convergence')
        else:
            print('Initial Model Converged. Proceeding to Next Step.')
            #Doing this solve command is required for GridPV, that is why the monitors
            #go under a reset process
            DSSMon.ResetAll()
            setSolutionParams(DSSObj,'daily',1,1,'off',1000000,30000)
            print('OpenDSS Reset Done.')
        setSourceInfo(DSSObj,['source'],'pu',[self.openDSSkwargs['SlackBusVoltage']])
        
        self.DSSObj = DSSObj
        self.DSSSolution = DSSSolution
    

    ########################### plot function ################################
    def plot_voltage_at_nodes(self, nodes, height=8, width=15):
        f = plt.figure()
        f.set_figheight(height)
        f.set_figwidth(width)

        plt.xlabel('Time Step')
        plt.ylabel('Voltage Unit')
        plt.title('Voltage Unit w.r.t Time Step', size=20)
        for node in nodes:
            plt.plot(self.nodes[node].loc['Voltage'], label=self.AllLoadNames[node]) 

        plt.legend()
        plt.show()

    def plot_y_at_nodes(self, nodes, height=8, width=15):
        f = plt.figure()
        f.set_figheight(height)
        f.set_figwidth(width)

        plt.xlabel('Time Step')
        plt.ylabel('Voltage Unit')
        plt.title('Y-value w.r.t Time Step', size=20)
        for node in nodes:
            plt.plot(self.inverters[node][0]['device'].y, label=self.AllLoadNames[node]) 

        plt.legend()
        plt.show() 

    def plot_VBP_at_nodes(self, nodes, height=8, width=15):
        f = plt.figure()
        f.set_figheight(height)
        f.set_figwidth(width)

        plt.xlabel('Time Step')
        plt.ylabel('Voltage Unit')
        plt.title('VBP w.r.t Time Step', size=20)
        for node in nodes:
            plt.plot(self.inverters[node][0]['controller'].VBP, label=self.AllLoadNames[node]) 

        plt.legend()
        plt.show() 