clc;
clear 
close all;

%% Load the components related to OpenDSS
[DSSObj, DSSText, gridpvpath] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;

%% Compile the Model, Directory of the OpenDSS file, Please find the associated Feeder 
% And Make sure you have the appropriate directory
DSSText.command = 'Compile C:\feeders\feeder13_B_R_Solar\feeder13_B_R_Solar.dss';
%% Solving the power flow in QSTS mode
setSolutionParams(DSSObj,'daily',1440,1,'static',1000,100);
DSSSolution.Solve();

%% Get all the Monitors
DSSMon=DSSCircuit.Monitors;
% Set this one as the active monitor
DSSMon.Name='solar_632';
fprintf('\nMonitor Name: %s\n',DSSMon.Name);
Headers=DSSMon.Header;
fprintf('Header Name: %s\n',Headers{:} );
time=ExtractMonitorData(DSSMon,0,1);
Pkw = ExtractMonitorData(DSSMon,1,1)+ExtractMonitorData(DSSMon,3,1)+ExtractMonitorData(DSSMon,5,1);
Qvar = ExtractMonitorData(DSSMon,2,1)+ExtractMonitorData(DSSMon,4,1)+ExtractMonitorData(DSSMon,6,1);
figure
plot(time,-Qvar,'r',time,-Pkw,'b','linewidth',1.5);
legend ('kVAR', 'kW')
xlim([1 length(time)]);
title('Power Supplied by the Substation')


%% Plotting the Voltage 
DSSMon.Name='VI_bus_692';
fprintf('\nMonitor Name: %s\n',DSSMon.Name);
Headers=DSSMon.Header;
fprintf('Header Name: %s\n',Headers{:} );
time=ExtractMonitorData(DSSMon,0,1);
Base_Voltage=4160/sqrt(3);
Vrms = ExtractMonitorData(DSSMon,1,Base_Voltage)+ExtractMonitorData(DSSMon,3,Base_Voltage)+ExtractMonitorData(DSSMon,5,Base_Voltage);
figure
plot(time,Vrms/3,'r','linewidth',1.5);
legend ('Voltage')
xlim([1 length(time)]);
ylabel('pu')
title('Voltage at Bus 692')

%% Get a specific XY Curve 
XYCurve=getXYCurveInfo(DSSObj,{'vv_curve_675'});
XYCurve.xarray=[0.5 0.95 1.05 1.3];
% Change the XY Curve
setXYCurveInfo(DSSObj,{'vv_curve_675'},XYCurve);
% Check whether the change occured or not
XYCurve=getXYCurveInfo(DSSObj,{'vv_curve_675'});


% Try to make a mistake to show whether the error gets generated
XYCurve.xarray=[0.5 0.95 1.05 1.3 1.5 ];
% The error is generated because the number of points does not match the
% npts property
setXYCurveInfo(DSSObj,{'vv_curve_675'},XYCurve);
