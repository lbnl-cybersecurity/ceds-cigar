from node import Node
from setInfo import *
from getInfo import *
from math import tan,acos

class Grid(object):
	_instance = None
	power_factor=0.9
	pf_converted=tan(acos(power_factor))

	def __new__(cls):
		if Grid._instance is None:
			Grid._instance = object.__new__(cls)
		return Grid._instance

	def set_name(cls, gridName):
		Grid._instance.gridName = gridName
		Grid._instance.nodes = {}

	def set_simulation_engine(cls, DSSObj, DSSSolution):
		Grid._instance.DSSObj = DSSObj
		Grid._instance.DSSSolution = DSSSolution

	def set_load_generation(cls, load, generation):
		Grid._instance.load = Grid._instance._preprocess_data(load)
		Grid._instance.generation = Grid._instance._preprocess_data(generation)

	def get_node(cls, nodeName):
		return Grid._instance.nodes[nodeName]
		
	def add_node(cls, nodeName):
		newNode = Node(Grid._instance, nodeName)
		Grid._instance.nodes[nodeName] = newNode

	def reset(cls):
		Grid._instance.timeStep = 0
		Grid._instance.terminal = False
		nodeNames = list(Grid._instance.nodes.keys())
		Grid._instance.invReal = {}
		Grid._instance.invReact = {}
		for node in nodeNames:
			Grid._instance.invReal[node] = 0
			Grid._instance.invReal[node] = 0 
			setLoadInfo(Grid._instance.DSSObj, [node], 'kw', [Grid._instance.load[node][Grid._instance.timeStep]])
			setLoadInfo(Grid._instance.DSSObj, [node], 'kvar', [Grid._instance.pf_converted*Grid._instance.load[node][Grid._instance.timeStep]])
		Grid._instance.DSSSolution.Solve()
		if (not Grid._instance.DSSSolution.Converged):
			print('Solution Not Converged at Step:', Grid._instance.timeStep)
		nodeInfo = getLoadInfo(Grid._instance.DSSObj, nodeNames)
		
		#parsing voltage for easy to use
		Grid._instance.voltage = {}
		for info in nodeInfo:
			Grid._instance.voltage[info['name']] = info['voltagePU']

		#making a state wraping here to broadcast to node
		package = {}
		for node in nodeNames:
			package[node] = {'voltage': Grid._instance.voltage[node], 
							'load': Grid._instance.load[node][Grid._instance.timeStep], 
							'generation': Grid._instance.generation[node][Grid._instance.timeStep],
							'terminal': Grid._instance.terminal}
		
		Grid._instance.grid_state = package
		Grid._instance.timeStep += 1

	def step(cls):
		#at first step, we are not injecting in anything, so basically it is like reset(), but we forward the Info to Node
		if Grid._instance.timeStep == 1:
			Grid._instance.terminal = False
			nodeNames = list(Grid._instance.nodes.keys())
			Grid._instance.invReal = {}
			Grid._instance.invReact = {}
			for node in nodeNames:
				Grid._instance.invReal[node] = 0
				Grid._instance.invReal[node] = 0 
				setLoadInfo(Grid._instance.DSSObj, [node], 'kw', [Grid._instance.load[node][Grid._instance.timeStep]])
				setLoadInfo(Grid._instance.DSSObj, [node], 'kvar', [Grid._instance.pf_converted*Grid._instance.load[node][Grid._instance.timeStep]])
			Grid._instance.DSSSolution.Solve()
			if (not Grid._instance.DSSSolution.Converged):
				print('Solution Not Converged at Step:', Grid._instance.timeStep)
			nodeInfo = getLoadInfo(Grid._instance.DSSObj, nodeNames)
		
			#parsing voltage for easy to use
			Grid._instance.voltage = {}
			for info in nodeInfo:
				Grid._instance.voltage[info['name']] = info['voltagePU']

			#making a state wraping here to broadcast to node
			package = {}
			for node in nodeNames:
				package[node] = {'voltage': Grid._instance.voltage[node], 
								'load': Grid._instance.load[node][Grid._instance.timeStep], 
								'generation': Grid._instance.generation[node][Grid._instance.timeStep],
								'terminal': Grid._instance.terminal}
		
			Grid._instance.grid_state = package
			Grid._instance.timeStep += 1



	#node observer
	def _notify(cls):
		for nodeName in Grid._instance.nodes:
			node = Grid._instance.nodes[nodeName]
			node.update(Grid._instance._grid_state)

	@property
	def grid_state(cls):
		return Grid._instance._grid_state

	@grid_state.setter
	def grid_state(cls, arg):
		Grid._instance._grid_state = arg
		Grid._instance._notify()


	def _preprocess_data(cls, data):
		newData = {}
		Grid._instance.terminal = data.shape[0]
		nodeNames = list(Grid._instance.nodes.keys())
		for i in range(len(nodeNames)):
			newData[nodeNames[i]] = data[:,i] 
		return newData
