clc;
clear;
close all;

%% Load the components related to OpenDSS
[DSSObj, DSSText, gridpvpath] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSMon=DSSCircuit.Monitors;
DSSText.command = 'Clear';

%% QSTS Simulation
DSSText.command = 'Compile C:\feeders\feeder34_B_NR\feeder34_B_NR.dss';
% Doing this solve command is required for GridPV, that is why the monitors
% go under a reset process
DSSSolution.Solve();
DSSMon.ResetAll;

TotalTimeSteps=1440;
setSolutionParams(DSSObj,'daily',1,1,'off',1000000,30000);
% Easy process to get all names and count of loads 
TotalLoads=DSSCircuit.Loads.Count;
AllLoadNames=DSSCircuit.Loads.AllNames;
% Creating a random distribution between 0 and 1, This part will get
% changed whien you will try to incorporate the actual load profile
rp = 0.95 + (1.05-0.95).*rand(TotalTimeSteps,1);
rq = 0.95 + (1.05-0.95).*rand(TotalTimeSteps,1);


Loads=getLoadInfo(DSSObj);
P=[Loads.kW];
Q=[Loads.kvar];
for i =1:TotalTimeSteps
    setLoadInfo(DSSObj,AllLoadNames,'kw',P*rp(i));
    setLoadInfo(DSSObj,AllLoadNames,'kvar',Q*rq(i));
    
    % All the control Code should go here
    DSSSolution.Solve();
end


%% Plotting Monitors Data

DSSMon.Name='utility';
fprintf('\nMonitor Name: %s\n',DSSMon.Name);
time=ExtractMonitorData(DSSMon,0,DSSSolution.StepSize);
Power= ExtractMonitorData(DSSMon,1,1);
Qvar = ExtractMonitorData(DSSMon,2,1);
figure
% Getting the Power from Substation 
plot(time,Power,'r',time,Qvar,'b','linewidth',1.5);
legend('Real Power','Reactive Power')
xlim([1 length(time)]);
title('Power From the Substation')

% BaseVoltage=24.9*1000/sqrt(3);
% DSSMon.Name='solar 01 VI';
% Voltage_01= ExtractMonitorData(DSSMon,1,BaseVoltage);
% figure
% plot(time,Voltage_01 / 3,'k', 'linewidth',1.5)
% xlim([1 length(time)]);
% title ('Bus Voltage (Average)')






