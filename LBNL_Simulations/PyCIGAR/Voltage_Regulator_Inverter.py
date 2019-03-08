# Importing the libraries
import matplotlib.pyplot as plt
import opendssdirect as dss
import numpy as np

dss.run_command("Redirect 34BusLTC/Radial34Bus.dss")

#  Enabling the controls for both regulators and Inverters; OpenDSS only allow to change the PVSystem Parameters, not the INvControl parameters
dss.Text.Command('BatchEdit PVSystem..* pctpmpp=100')
dss.Text.Command('BatchEdit InvControl..* enabled=Yes')
dss.Text.Command('BatchEdit RegControl..* enabled= Yes')

dss.RegControls.Name('ltc-t_02')
# Set the max tap change to be 1
dss.RegControls.MaxTapChange(1)
# Run a Power Flow Solution
dss.Solution.Solve()

# Extract the monitor data
dss.Monitors.Name('tapMonitor')
time = dss.Solution.Number() / len(dss.Monitors.dblHour()) * 60 * np.asarray(dss.Monitors.dblHour())  # Because the simulation uses step size of one minute and one hour = 60 minute
tap = np.asarray(dss.Monitors.Channel(1))

dss.Monitors.Name('solar 01')
real_power = np.asarray(dss.Monitors.Channel(1)) + np.asarray(dss.Monitors.Channel(3)) + np.asarray(dss.Monitors.Channel(5))
reactive_power = np.asarray(dss.Monitors.Channel(2)) + np.asarray(dss.Monitors.Channel(4)) + np.asarray(dss.Monitors.Channel(6))

base_voltage = 24.9 * 1000 / (3 ** 0.5)
dss.Monitors.Name('solar 01 VI')
voltage = np.asarray(dss.Monitors.Channel(1)) + np.asarray(dss.Monitors.Channel(3)) + np.asarray(dss.Monitors.Channel(5))
# Grab the monitor data 
print(dss.Monitors.dblHour()[-1])

# The section for plotting data
plt.figure()
plt.plot(time, tap, label='Transformer Tap')
plt.title('Transformer Tap Position')
plt.xlabel('minutes')
plt.legend()
plt.show()

plt.figure()
plt.plot(time, -real_power, label='Real Power')  # Because generation is negative
plt.plot(time, -reactive_power, label='Reactive Power')
plt.title('Real and Reactive Power from Solar ')
plt.xlabel('minutes')
plt.legend()
plt.show()

plt.figure()
plt.plot(time, voltage / 3 / base_voltage, label='Voltage')
plt.title('Average Per Unit Bus Voltage')
plt.xlabel('minutes')
plt.legend()
plt.show()
