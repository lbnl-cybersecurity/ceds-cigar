from controller_inverter import Inverter
class Node(object):
	_controllers = []
	
	def __init__(self, grid, nodeName):
		self.grid = grid
		self.nodeName = nodeName

	def add_controller(self, **args):
		if 'controller_type' in args:
			controller_type = args['controller_type']
			if controller_type == 'inverter':
				newController = Inverter(self, sbar=args['sbar'])
		
		if 'status' in args:
			status = status
		else:
			status = 'unhacked'	

		newControl = {"controller": newController, 
						"controller_type": controller_type,
						"status": status
						}
		
		self._controllers.append(newControl)

	def get_controllers(self):
		return self._controllers

	def update(self, package):
		self.voltage = package[self.nodeName]['voltage']
		#print(self.voltage)

