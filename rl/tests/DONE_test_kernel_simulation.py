from pycigar.core.kernel.kernel import Kernel

k = Kernel(simulator="opendss", sim_params="ahihi")

kernel_api = k.simulation.start_simulation("a")

command = "Redirect tests/feeder34_test.dss"
kernel_api.simulation_command(command)
