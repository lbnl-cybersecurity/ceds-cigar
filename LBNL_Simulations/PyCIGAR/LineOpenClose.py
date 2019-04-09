'''
This code is an example of how to open and close lines in OpenDSS using the DSSText Command.
'''

import opendssdirect as dss

dss.run_command('Redirect feeder/33BusMeshed/33BusMeshed.dss')
dss.Solution.Solve()
dss.Circuit.SetActiveElement('line_30')
print('Flow Before Opening  line 29 and 5: {}'.format(dss.CktElement.Powers()))

# Opening two lines; this is the easiest way of doing this without going through the setting function
# remember the enabled feature does not work as it should be 
# Follow this discussion form if required : https://sourceforge.net/p/electricdss/discussion/861976/thread/7c3a789530/ 
dss.Text.Command('open line.line_29 term=1')
dss.Text.Command('open line.line_5 term=1')

dss.Solution.Solve()
dss.Circuit.SetActiveElement('line_30')
print('Flow After Opening  line 29 and 5: {}'.format(dss.CktElement.Powers()))
# The values found here are close because the network is weakly msshed; in case of a radial network the branches underlying will have zero current

#  Close the two lines 
dss.Text.Command('close line.line_29 term=1')
dss.Text.Command('close line.line_5 term=1')

# Check whether the flows restored
dss.Solution.Solve()
dss.Circuit.SetActiveElement('line_30')
print('Flow After Closing line 29 and 5: {}'.format(dss.CktElement.Powers()))
