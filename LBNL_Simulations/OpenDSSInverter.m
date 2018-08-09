% Author : Shammya Saha (sssaha@lbl.gov)
% This function demonstrates the combined control action of PV Controls,  Regulator Controls, and Cap controls  
clc;
close all;
clear
%% OpenDSS integration

[DSSObj, DSSText, gridpvpath] = DSSStartup;
% Load the components related to OpenDSS
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;

%Compile the Model, Directory of the OpenDSS file, Please find the associated Feeder 
% And Make sure you have the appropriate directory
DSSText.command = 'Compile C:\feeders\feeder13_U_R_Pecan_Solar\feeder13_U_R_Pecan_Solar.dss';
%Turning off the Regulator Controls and Cap Controls
% This is the fastest way of getting all the names of the cap controls
% setRegInfo(DSSObj,DSSCircuit.RegControls.AllNames,'enabled',zeros(1,length(DSSCircuit.RegControls.AllNames)));
% setCapControlInfo(DSSObj,DSSCircuit.CapControls.AllNames,'enabled',zeros(1,length(DSSCircuit.CapControls.AllNames)));


%% Solving the power flow in QSTS mode
% % DSSSolution.ControlMode=0;
DSSSolution.Mode=1; % 1 represents daily mode, 2 represents yearly mode
DSSSolution.Number=1440; % Solutions Per Solve Command
DSSSolution.StepSize=1; % Stepsize= 1s
DSSSolution.MaxControlIterations=1000; % Increase the number of maximum control iterations to make sure the system can solve the power flow
DSSSolution.MaxIterations=100; % Increasing the number of power flow iterations to achieve convergence
% Set Appropriate Control Mode
% {OFF | STATIC |EVENT | TIME}  Default is "STATIC".  
% Control mode for the solution. Set to OFF to prevent controls from changing.
% STATIC = Time does not advance.  Control actions are executed in order of shortest time to act until 
% all actions are cleared from the control queue.  Use this mode for power flow solutions which may require several regulator tap changes per solution.
% EVENT = solution is event driven.  Only the control actions nearest in time are executed and the time 
% is advanced automatically to the time of the event. 
% TIME = solution is time driven.  Control actions are executed when the time for the 
% pending action is reached or surpassed.
% Controls may reset and may choose not to act when it comes their time. 
% Use TIME mode when modeling a control externally to the DSS 
% and a solution mode such as DAILY or DUTYCYCLE that advances time, or set the time (hour and sec) explicitly from the external program. 
DSSText.Command='Set ControlMode=EVENT';
DSSSolution.Solve();

%% This section of the code actually shows how to do the resolve of the network
% first do the cleanup so the history of the previous solution is completey
% wiped
DSSSolution.Cleanup; 
DSSText.command = 'Compile C:\feeders\feeder13_U_R_Pecan_Solar\feeder13_U_R_Pecan_Solar.dss';
% DSSSolution.Mode=1; % 1 represents daily mode, 2 represents yearly mode
% DSSSolution.Number=1440; % Solutions Per Solve Command
% DSSSolution.StepSize=1; % Stepsize= 1s
% DSSSolution.MaxControlIterations=1000; % Increase the number of maximum control iterations to make sure the system can solve the power flow
% DSSSolution.MaxIterations=100;
% DSSText.Command='Set ControlMode=Time';
setSolutionParams(DSSObj,'daily',1440,1,'OFF')
DSSSolution.Solve();

%% PLotting the values 
% Plotting the real and reactive power taken from the substation
TimeNormalizingFactor=1;
DSSMon=DSSCircuit.Monitors;
DSSMon.Name='Meter_632_power';
fprintf('\nMonitor Name: %s\n',DSSMon.Name);
Headers=DSSMon.Header;
fprintf('Header Name: %s\n',Headers{:} );
Pkw = ExtractMonitorData(DSSMon,1,1)+ExtractMonitorData(DSSMon,3,1)+ExtractMonitorData(DSSMon,5,1);
Qvar = ExtractMonitorData(DSSMon,2,1)+ExtractMonitorData(DSSMon,4,1)+ExtractMonitorData(DSSMon,6,1);
time=ExtractMonitorData(DSSMon,0,TimeNormalizingFactor); % This need to be done to convert the time axis to hours 
figure
% negative Value as generation is negative by default in OpenDSS
subplot(211)
plot(time,-Qvar,'linewidth',1.5);
xlim([1 length(time)]);
title('VAR Taken From Solar at Bus 632')

subplot(212)
plot(time,-Pkw,'linewidth',1.5);
xlim([1 length(time)]);
title('Real Power Taken From Solar at Bus 632')

% Plotting power from the solar at bus 632
DSSMon=DSSCircuit.Monitors;
DSSMon.Name='Meter_692_power';
fprintf('\nMonitor Name: %s\n',DSSMon.Name);
Headers=DSSMon.Header;
fprintf('Header Name: %s\n',Headers{:} );
Pkw = ExtractMonitorData(DSSMon,1,1)+ExtractMonitorData(DSSMon,3,1)+ExtractMonitorData(DSSMon,5,1);
Qvar = ExtractMonitorData(DSSMon,2,1)+ExtractMonitorData(DSSMon,4,1)+ExtractMonitorData(DSSMon,6,1);
time=ExtractMonitorData(DSSMon,0,TimeNormalizingFactor); % This need to be done to convert the time axis to hours 
figure
% negative Value as generation is negative by default in OpenDSS
subplot(211)
plot(time,-Qvar,'linewidth',1.5);
xlim([1 length(time)]);
title('VAR Taken From Solar at Bus 692')

subplot(212)
plot(time,-Pkw,'linewidth',1.5);
xlim([1 length(time)]);
title('Real Power Taken From Solar at Bus 692')