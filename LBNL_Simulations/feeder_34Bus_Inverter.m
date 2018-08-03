clc;
clear;
close all;

%% TUNING KNOBS - ADJUST (MAIN) PARAMETERS HERE 
% Note that there are other tunable parameters in the code, but these are
Sbase=1;
LoadScalingFactor = 2000; 
GenerationScalingFactor = 70; 
SlackBusVoltage = 1.02; 

% Set simulation analysis period
StartTime = 40000; 
EndTime = 40500; 

% Set hack parameters
TimeStepOfHack = 50;
PercentHacked = [0 0 1.0 0 0 0.5 0.5 0.5 0 0.5 0.5 0.5 0.5,0];

% Set initial VBP parameters for uncompromised inverters
VQ_start = 1.01; VQ_end = 1.03; VP_start = 1.03; VP_end = 1.05;

% Set VBP parameters for hacked inverters (i.e. at the TimeStepOfHack,
% hacked inverters will have these VBP)
VQ_startHacked = 1.01; VQ_endHacked = 1.015; VP_startHacked = 1.015; VP_endHacked = 1.02;

% Set adaptive controller gain values (the higher the gain, the faster the
% response time)
kq = 1;
kp = 1;

% Set delays for each node
Delay_VoltageSampling = [0 0  10 0 0  10  10  50  10  10  10  10  10]; 
Delay_VBPCurveShift =   [0 0 120 0 0 120 120 120 120 120 120 120 120]; 

% Set observer voltage threshold
ThreshHold_vqvp = 0.25;
power_factor=0.9;
pf_converted=tan(acos(power_factor));



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
setSolutionParams(DSSObj,'daily',1,1,'off',1000000,30000);
% Easy process to get all names and count of loads 
TotalLoads=DSSCircuit.Loads.Count;
AllLoadNames=DSSCircuit.Loads.AllNames;

%% Retrieving the data from the load profile
TimeResolutionOfData=10; % resolution in minute
% Get the data from the Testpvnum folder
% Provide Your Directory
FileDirectoryBase='C:\feeders\testpvnum10\';
QSTS_Time = 0:1440; % This can be changed based on the available data
TotalTimeSteps=length(QSTS_Time);
QSTS_Data = zeros(length(QSTS_Time),4,TotalLoads); % 4 columns as there are four columns of data available in the .mat file


for node = 1:TotalLoads
    % This is created manually according to the naming of the folder
    FileDirectoryExtension= strcat('node_',num2str(node),'_pv_',num2str(TimeResolutionOfData),'_minute.mat');
    % The total file directory
    FileName=strcat(FileDirectoryBase,FileDirectoryExtension);
    % Load the 
    MatFile = load(FileName,'nodedata');    
    QSTS_Data(:,:,node) = MatFile.nodedata; %Putting the loads to appropriate nodes according to the loadlist
end

Generation = QSTS_Data(:,2,:)*GenerationScalingFactor; % solar generation
Load = QSTS_Data(:,4,:)*LoadScalingFactor; % load demand

Generation=squeeze(Generation)/Sbase; % To convert to per unit, it should not be multiplied by 100
Load=squeeze(Load)/Sbase; % To convert to per unit
MaxGenerationPossible = max(Generation); % Getting the Maximum possible Generation for each Load 

%% Interpolate to change data from minutes to seconds
Time = StartTime:EndTime;
TotalTimeSteps=length(Time);

% Interpolate to get minutes to seconds
for i = 1:TotalLoads
    t_seconds = linspace(1,numel(Load(:,i)),3600*24/1);
    LoadSeconds(:,i) = interp1(Load(:,i),t_seconds,'spline');
    GenerationSeconds(:,i)= interp1(Generation(:,i),t_seconds,'spline');
end

% Initialization
LoadSeconds = LoadSeconds(StartTime:EndTime,:);
GenerationSeconds = GenerationSeconds(StartTime:EndTime,:);
Load = LoadSeconds;
Generation = GenerationSeconds;

% Create noise vector
for node_iter = 1:TotalLoads
    Noise(:,node_iter) = randn(TotalTimeSteps, 1);
end 

% Add noise to loads
for i = 1:TotalLoads
    Load(:,i) = Load(:,i) + 2*Noise(:,i);
end 



%% Initializing the inverter models 
Number_of_Inverters = 13;
if Number_of_Inverters> TotalLoads
    exit('Not Supported Right now');
end
% Creating an object of array
InverterArray(1:Number_of_Inverters)=Inverter;
for i = 1:Number_of_Inverters
    InverterArray(i).Delay_VoltageSampling=10;
    InverterArray(i).Delay_VBPCurveShift=120;
    InverterArray(i).LPF=1;
    InverterArray(i).LowPassFilterFrequency=0.1;
    InverterArray(i).HighPassFilterFrequency=1;
    InverterArray(i).Gain_Energy=1e5;
    InverterArray(i).TimeStep=1;
    InverterArray(i).kp=1;
    InverterArray(i).kq=1; 
    InverterArray(i).ThreshHold_vqvp=0.25; % observer Threshhold
    InverterArray(i).PercentHacked=PercentHacked(i); % percent hacked
    InverterArray(i).ROC_lim=10; % currently unused
    InverterArray(i).InverterRateOfChangeActivate=0; % currently unused 
end

%% Setting up the maximum power capability
Sbar_max =  MaxGenerationPossible * GenerationScalingFactor;
Sbar = zeros(size(Generation));
SbarHacked = Sbar_max .* PercentHacked;

for t = 1:TotalTimeSteps
    Sbar(t,:) = Sbar_max;
end

for t = TimeStepOfHack:TotalTimeSteps
    Sbar(t,:) = Sbar_max.*(1-PercentHacked);
end 




% Initialize VBP for hacked and uncompromised inverters
VBP = [nan*ones(IeeeFeeder,1,TotalTimeSteps), nan*ones(IeeeFeeder,1,TotalTimeSteps), ...
       nan*ones(IeeeFeeder,1,TotalTimeSteps), nan*ones(IeeeFeeder,1,TotalTimeSteps)];
VBPHacked = VBP;

% Hard-code initial VBP points
VBP(:,1,1:2) = VQ_start;
VBP(:,2,1:2) = VQ_end;
VBP(:,3,1:2) = VP_start;
VBP(:,4,1:2) = VP_end;
VBPHacked(:,1,1:2) = VQ_startHacked;
VBPHacked(:,2,1:2) = VQ_endHacked;
VBPHacked(:,3,1:2) = VP_startHacked;
VBPHacked(:,4,1:2) = VP_endHacked;






%% Voltage Observer Parameters and related variable initialization

InverterReactivePower = zeros(size(Generation));
InverterRealPower = zeros(size(Generation));
InverterReactivePowerHacked = zeros(size(Generation));
InverterRealPowerHacked = zeros(size(Generation));
FilteredOutput_vqvp = zeros(TotalTimeSteps,NumberOfNodes);
IntermediateOutput_vqvp=zeros(TotalTimeSteps,NumberOfNodes);
Epsilon_vqvp = zeros(TotalTimeSteps,NumberOfNodes);
upk = zeros(size(IntermediateOutput_vqvp));
uqk = upk;
FilteredVoltage = zeros(size(Generation));
FilteredVoltageCalc = zeros(size(Generation));
%% OpenDSS Parameters

for ksim =1:TotalTimeSteps
    %setLoadInfo(DSSObj,AllLoadNames,'kw',(Load(ksim,LoadList)-SolarGeneration_NC(ksim,LoadList))/1000); % To convert to KW
    if (ksim>1)
        setLoadInfo(DSSObj,AllLoadNames,'kw',(Load(ksim,:)+InverterRealPower(ksim-1,:) + InverterRealPowerHacked(ksim-1,:))); % To convert to KW
        setLoadInfo(DSSObj,AllLoadNames,'kvar',pf_converted*(Load(ksim,:))+InverterReactivePower(ksim-1,knode) +InverterReactivePowerHacked(ksim-1,knode));
    else
        setLoadInfo(DSSObj,AllLoadNames,'kw',(Load(ksim,:))); % To convert to KW
        setLoadInfo(DSSObj,AllLoadNames,'kvar',pf_converted*(Load(ksim,:)));
  
    end
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







