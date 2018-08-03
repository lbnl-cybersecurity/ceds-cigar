clc;
clear;
% close all;

%% Load the components related to OpenDSS
[DSSObj, DSSText, gridpvpath] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSText.command = 'Clear';

%% QSTS Simulation
DSSText.command = 'Compile C:\feeders\34BusLTCInverse\Radial34BusLTCInverse.dss';
setSolutionParams(DSSObj,'daily',1440,1,'Time',1000000,30000);
DSSSolution.Solve();


%% Plotting Monitors Data
DSSMon=DSSCircuit.Monitors;
DSSMon.Name='tapMonitor';
fprintf('\nMonitor Name: %s\n',DSSMon.Name);
Headers=DSSMon.Header;
fprintf('Header Name: %s\n',Headers{:} );
time=ExtractMonitorData(DSSMon,0,DSSSolution.StepSize);
tap= ExtractMonitorData(DSSMon,1,1);
figure
plot(time,tap,'r','linewidth',1.5);
legend('Tap Position (pu)')
xlim([1 length(time)]);

DSSMon=DSSCircuit.Monitors;
DSSMon.Name='tapvoltage';
fprintf('\nMonitor Name: %s\n',DSSMon.Name);
Headers=DSSMon.Header;
fprintf('Header Name: %s\n',Headers{:} );
time=ExtractMonitorData(DSSMon,0,DSSSolution.StepSize);
Voltage= ExtractMonitorData(DSSMon,1,24.9/sqrt(3)*1000)+...
    ExtractMonitorData(DSSMon,3,24.9/sqrt(3)*1000)+...
    ExtractMonitorData(DSSMon,5,24.9/sqrt(3)*1000);
figure
plot(time,Voltage/3,'r','linewidth',1.5);
legend('Tap Position (pu)')
xlim([1 length(time)]);

plot(time,Voltage/3,'r',time,tap,'k','linewidth',1.5);
xlim([1 length(time)]);

DSSText.command = 'Show EventLog';



