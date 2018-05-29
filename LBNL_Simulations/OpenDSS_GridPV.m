%This code actually shows the capability of GridPV model
% This code will be updated peiodically for further development
% Please refer to GridPV documentation for further details 
% This code also illustrates some of the set functions developed

clc;
clear;
close all;

%% Variables related to OpenDSS
[DSSObj, DSSText, gridpvpath] = DSSStartup;
% Load the components related to OpenDSS
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSMon=DSSCircuit.Monitors;

%% Compile the Model
DSSText.command = 'Compile C:\feeders\feeder13_B_R\feeder13BR.dss';
DSSSolution.Solve();


%% Following Code shows some example for some components, how to use GridPV
LoadInfo = getLoadInfo(DSSObj, [{'load_611'};{'load_645'}]);
LoadInfo_kW=[LoadInfo.kW];
LoadInfo_absorbedpower=[LoadInfo.powerReal];


LineInfo = getLineInfo(DSSObj, [{'L_U_650'};{'L_632_645'}]);
IncomingRealPower=[LineInfo.bus1PowerReal];
IncomingReactivePower=[LineInfo.bus1PowerReactive];
disp(IncomingRealPower)

BusInfo = getBusInfo(DSSObj, {'bus_634'});
VoltagePU=[BusInfo.voltagePU];
%% Change Load Parameter

setLoadInfo(DSSObj,[{'load_645'},{'load_611'}],'kw',[200,300]);
%Resolve Power Flow
DSSSolution.Solve();
% You need to read the line data again as OpenDSS does not provide direct
% access to memory
LineInfo = getLineInfo(DSSObj, [{'L_U_650'};{'L_632_645'}]);
IncomingRealPower=[LineInfo.bus1PowerReal];
IncomingReactivePower=[LineInfo.bus1PowerReactive];
disp(IncomingRealPower)

%% Get Loss Value
Losses= DSSCircuit.LineLosses;

%% Change Source Voltage
setSourceInfo(DSSObj,[{'source'}],'pu',[1.00]);
DSSSolution.Solve();
BusInfo=getBusInfo(DSSObj);
Busvoltage=[BusInfo.voltagePU]

setSourceInfo(DSSObj,[{'source'}],'pu',[1.02]);
DSSSolution.Solve();
BusInfo=getBusInfo(DSSObj);
Busvoltage=[BusInfo.voltagePU]























%DSSCircuit.SetActiveElement('Line.L_U_650')
%DSSCircuit.ActiveElement.Powers
% Power=ExtractMonitorData(DSSMon,1,1)

% Loads = getLoadInfo(DSSCircObj);
% Loads = getLoadInfo(DSSCObj, [{'load_611'};{'load_645'}]);

% DSSCircuit = DSSObj.ActiveCircuit;
% Loads=DSSCircuit.Loads;
% AllLoadNames=DSSCircuit.Loads.AllNames;
% for i=1:Loads.Count
% %     if (~strcmpi(AllLoadNames{i},'load_700'))
% %         error('No Load Found');
% %     end
% % end
% % loadinfo=getLoadInfo(DSSObj);
%   DSSObj=setLoadInfo(DSSObj,[{'load_645'},{'load_611'}],'kw',[200,300]);
%   DSSSolution.Solve;
%  % Getting Powers for a line 
%  DSSCircuit.SetActiveElement('Line.L_U_650')
% DSSCircuit.ActiveElement.Powers
% DSSCircuit.AllBusVmagPu
% DSSMon=DSSCircuit.Monitors;
% DSSMon.Name='M_L_632_645';
% Power=ExtractMonitorData(DSSMon,1,1)