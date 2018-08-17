from grid import Grid 
import main
grid = Grid()

grid.set_name("feeder_34Bus")

nodeNames = ['dl_82034','dl_858m','dload_806m',
'dload_810', 'dload_810m','dload_822','dload_822m', 
'dload_824','dload_826','dload_826m','dload_828', 
'dload_830','dload_830m','dload_834','dload_836', 
'dload_838','dload_838m','dload_840','dload_844',
'dload_846','dload_848','dload_856','dload_856m', 
'dload_860','dload_864','dload_864m','sload_840', 
'sload_844','sload_848','sload_860','sload_890']

#add node
for nodeName in nodeNames:
	grid.add_node(nodeName)

DSSObj, DSSSolution, load, generation, sbar = main.Main()

#add OpenDSS engine
grid.set_simulation_engine(DSSObj, DSSSolution)
#add scenario
grid.set_load_generation(load, generation)
##### DONE WITH GRIDS AND NODES ######

grid.reset()
#add inverters for each of the node
for node in nodeNames:
	grid.get_node(node).add_controller(controller_type='inverter', sbar=sbar[node])

print(grid.get_node('dl_82034').get_controllers()[0]['controller'])


