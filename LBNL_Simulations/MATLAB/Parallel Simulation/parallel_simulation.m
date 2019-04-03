% Author : Shammya Saha
% sssaha@lbl.gov / sssaha@asu.edu
% A sample code created to test parallel simulation capability of OpenDSS
clc;
clear;
close all;

%% Load the components related to OpenDSS
[DSSObj, DSSText, gridpvpath] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSText.command = 'clear all';

DSSText.command = 'set parallel=no';
% starts with actor 1
DSSText.command = 'compile (C:\ceds-cigar\LBNL_Simulations\MATLAB\Parallel Simulation\Radial34Bus.DSS)';
DSSText.command = 'set CPU=0';
DSSText.command = 'solve';
DSSText.command = 'NewActor';
DSSText.command = 'compile (C:\ceds-cigar\LBNL_Simulations\MATLAB\Parallel Simulation\33BusMeshed.DSS)';
DSSText.command = 'set CPU=1';
DSSText.command = 'solve';
DSSText.command = 'set parallel=yes';
DSSText.command = 'SolveAll';
DSSText.command = 'Wait';
DSSText.command = 'set ConcatenateReports=Yes';
DSSText.command = 'set activeactor=1'; % activates actor 1
DSSText.command = 'show voltages';
DSSText.command = 'set activeactor=2'; % activates actor 1
DSSText.command = 'show voltages';




