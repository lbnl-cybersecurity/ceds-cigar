clc;
clear;
close all;

%% Load the components related to OpenDSS
[DSSObj, DSSText, gridpvpath] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSText.command = 'Clear';

%% QSTS Simulation
DSSText.command = 'Compile C:\feeders\34BusLTC\Radial34Bus.dss';
% DSSText.command ='New XYCurve.vw_curve npts=4 Yarray = (1,1,0,0) XArray= (0.0,1.0,1.05,1.5)';
% DSSText.command ='New InvControl.InvPVCtrl mode=VOLTVAR voltage_curvex_ref=rated  vvc_curve1=vv_curve EventLog=yes  deltaQ_factor=0.3 VarChangeTolerance=0.05   VoltageChangeTolerance=0.001 VV_RefReactivePower=VARAVAL_WATTS';
% DSSText.command ='New InvControl.InvPVCtrl Combimode = VV_VW voltage_curvex_ref= rated vvc_curve1= vv_curve VV_RefReactivePower=VARAVAIL_WATTS EventLog= yes voltwatt_curve=vw_curve VoltageChangeTolerance=0.001 deltaP_factor=0.5';
% DSSText.command ='New InvControl.InvPVCtrl mode=VOLTWATT voltage_curvex_ref=rated  EventLog=yes  deltaP_factor=0.7 VarChangeTolerance=0.5  VoltageChangeTolerance=0.1 VV_RefReactivePower=VARAVAL_WATTS voltwatt_curve=vw_curve';

% DSSText.command='BatchEdit PVSystem..* pctpmpp=100';
% DSSText.command='BatchEdit InvControl..* enabled=Yes';
% DSSText.command='BatchEdit RegControl..* enabled= Yes';


setSolutionParams(DSSObj,'daily',1440,60,'static',1000000,30000);
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

DSSMon.Name='solar 01';
fprintf('\nMonitor Name: %s\n',DSSMon.Name);
time=ExtractMonitorData(DSSMon,0,DSSSolution.StepSize);
Power= ExtractMonitorData(DSSMon,1,1)+ExtractMonitorData(DSSMon,3,1)+ExtractMonitorData(DSSMon,5,1);
Qvar = ExtractMonitorData(DSSMon,2,1)+ExtractMonitorData(DSSMon,4,1)+ExtractMonitorData(DSSMon,6,1);
figure
plot(time,-Power/3,'r',time,-Qvar/3,'b','linewidth',1.5);
legend('Real Power','Reactive Power')
xlim([1 length(time)]);
title('Solar Power')

BaseVoltage=24.9*1000/sqrt(3);
DSSMon.Name='solar 01 VI';
Voltage_01= ExtractMonitorData(DSSMon,1,BaseVoltage)+ExtractMonitorData(DSSMon,3,BaseVoltage)+ExtractMonitorData(DSSMon,5,BaseVoltage);
figure
plot(time,Voltage_01 / 3,'k', 'linewidth',1.5)
xlim([1 length(time)]);
title ('Bus Voltage (Average)')
%%
% close all
% voltwatt=load('voltwatt.mat');
% voltvar=load('voltvar.mat');
% 
% time=voltwatt.time;
% tap_voltwatt=voltwatt.tap;
% tap_voltvar=voltvar.tap;
% figure
% plot(time,tap_voltwatt,'r',time,tap_voltvar,'b','linewidth',1.5);
% legend('VoltWatt Control','Voltvar Control');
% xlim([1 length(time)]);
% ylabel('Tap Position (pu)')
% 
% P_voltwatt=voltwatt.Power;
% P_voltvar=voltvar.Power;
% figure
% plot(time,-P_voltwatt,'r',time,-P_voltvar,'b','linewidth',1.5);
% legend('VoltWatt Control','Voltvar Control');
% xlim([1 length(time)]);
% ylabel('Real Power (kW)')
% 
% Q_voltwatt=voltwatt.Qvar;
% Q_voltvar=voltvar.Qvar;
% figure
% plot(time,-Q_voltwatt,'r',time,-Q_voltvar,'b','linewidth',1.5);
% legend('VoltWatt Control','Voltvar Control');
% xlim([1 length(time)]);
% ylabel('Rective Power (kVAR)')





