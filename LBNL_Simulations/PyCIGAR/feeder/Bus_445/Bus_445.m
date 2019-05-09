clc
clear all 

%% Variables related to OpenDSS
[DSSObj, DSSText, gridpvpath] = DSSStartup;
% Load the components related to OpenDSS
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSMon=DSSCircuit.Monitors;

%% Compile the Model
DSSText.command = 'Compile Bus_445.dss';
DSSSolution.Solve();

%%
figure;
plotCircuitLines(DSSObj)