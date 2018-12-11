from abc import ABCMeta, abstractmethod

class _controller(object):

	__metaclass__ = ABCMeta

	#which node this controller connected to

	def __init__(self, node, controller_type):
		self.node = node
		self.controller_type = controller_type

	def get_node(self):
		return self.node

	def set_node(self, node):
		self.node = node