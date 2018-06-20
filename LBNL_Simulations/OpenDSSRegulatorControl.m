% Author : Shammya Saha (sssaha@lbl.gov)
 
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
DSSText.command = 'Compile C:\feeders\feeder13_U_R_Pecan\feeder13_U_R_Pecan.dss';
DSSText.command = 'RegControl.LVR-lvr_03.debugtrace=yes';
% Changing the Max tap just to make sure this info can be retrieved later
DSSText.Command='RegControl.lvr-lvr_01.maxtapchange=10' ;
% Solving the power flow in QSTS mode
DSSText.command ='Solve mode="daily" number=1440 stepsize="1s"';
% The following line exports the monitor data in the CSV file, can be
% commented out if required
%DSSText.command ='Export monitor meter_634';
% Getting the Regulator Info
regs=getRegInfo(DSSObj);
fprintf('MaxTapChange: %d\n', regs.MaxTapChange);
% DSSSolution.Solve();
% Get all the Monitors
DSSMon=DSSCircuit.Monitors;
% Set this one as the active monitor
DSSMon.Name='meter_634';
% print the monitor name
fprintf('Monitor Name: %s\n',DSSMon.Name);
Headers=DSSMon.Header;
fprintf('Header Name: %s\n',Headers{:} );
% Getting the value in pu, 2400 is the base value of that bus
Vrms = ExtractMonitorData(DSSMon,1,2400);
time=ExtractMonitorData(DSSMon,0,1);
plot(time,Vrms,'linewidth',1.5);
xlabel('Time')
ylabel('Voltage (pu)')
title ('Voltage Recorded at Meter 634');
% DSSSolution.SampleControlDevices();
% DSSCircuit.CtrlQueue.Show();
% disp(DSSText.Result)



