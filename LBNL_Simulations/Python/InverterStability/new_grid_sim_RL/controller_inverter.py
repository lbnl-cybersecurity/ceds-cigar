from controller import _controller

class Inverter(_controller):
	controller_type = 'inverter'
	
	def __init__(self, node, **args):
		self.node = node
		self.sbar = args['sbar']

	