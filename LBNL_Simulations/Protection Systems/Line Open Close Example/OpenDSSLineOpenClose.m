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

DSSText.command = 'Compile 33BusMeshed.dss';

DSSSolution.Solve();
DSSCircuit.SetActiveElement('line_30');
% print the power values before opening the lines
disp(DSSCircuit.ActiveElement.Powers(1:2))

DSSText.command = 'open line.line_29 term=1';
DSSText.command = 'open line.line_5 term=1';

DSSSolution.Solve();
DSSCircuit.SetActiveElement('line_30');
% print the power values after opening the lines
disp(DSSCircuit.ActiveElement.Powers(1:2))

DSSText.command = 'close line.line_29 term=1';
DSSText.command = 'close line.line_5 term=1';

DSSSolution.Solve();
DSSCircuit.SetActiveElement('line_30');
% print the power values after re-closing-- the line 
disp(DSSCircuit.ActiveElement.Powers(1:2))

