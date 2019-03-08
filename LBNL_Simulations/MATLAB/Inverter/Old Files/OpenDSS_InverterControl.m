% Author : Daniel Arnold (dbarnold@lbl.gov)
% Date : 06/29/2018
% This code is adapted from OpenDSSvsFBSWithControl.m that Shammo wrote in
% May 2018

% June 29 - adapt code to create instability

clc;
clear;
close all;

%% Constants 
% These constants are used to create appropriate simulation scenario
LoadScalingFactor=0.9*1.5;
GenerationScalingFactor=2;
SlackBusVoltage=1.02;
power_factor=0.9;
IncludeSolar=1; % Setting this to 1 will include the solar , set to zero if no solar required

%Feeder parameters

LineNames = {'l_632_633'
    'l_632_645'
    'l_632_671'
    'l_633_634'
    'l_645_646'
    'l_650_632'
    'l_671_680'
    'l_671_684'
    'l_671_692'
    'l_684_611'
    'l_684_652'
    'l_692_675'
    'l_u_650'};

AllBusNames = { 'sourcebus'
    'load_611'
    'load_634'
    'load_645'
    'load_646'
    'load_652'
    'load_671'
    'load_675'
    'load_692'
    'bus_611'
    'bus_634'
    'bus_645'
    'bus_646'
    'bus_652'
    'bus_671'
    'bus_675'
    'bus_692'
    'bus_632'
    'bus_633'
    'bus_650'
    'bus_680'
    'bus_684'};

LoadBusNames = AllBusNames(2:9);
BusNames = AllBusNames(10:22);
IeeeFeeder = 13;

LoadList = [6,7,8,13,3,12,11,10];
NodeList = [650,632,671,680,633,634,645,646,684,611,692,675,652];
BusesWithControl = NodeList(LoadList);
LoadBusNames_ControlIndex = zeros(length(BusesWithControl),1);

for i=1:length(LoadBusNames_ControlIndex)
    %find the index in LoadBusNames where BusesWithControl(i) occurs
    LoadBusNames_ControlIndex(i) = find( contains( ...
        LoadBusNames,num2str(BusesWithControl(i)) ) );
end

NumberOfLoads=length(LoadBusNames);
NumberOfNodes=length(BusNames);

%% Base Value Calculation

Vbase = 4.16e3; %4.16 kV
Sbase = 1; %500 kVA

Zbase = Vbase^2/Sbase;
Ibase = Sbase/Vbase;

%% Load Data Pertaining to Loads to create a profile
TimeResolutionOfData=10; % resolution in minute
% Get the data from the Testpvnum folder
% Provide Your Directory
FileDirectoryBase = [pwd, '\testpvnum10\'];
Time = 0:1440; % This can be changed based on the available data
TotalTimeSteps=length(Time);
QSTS_Data = zeros(length(Time),4,IeeeFeeder); % 4 columns as there are four columns of data available in the .mat file

for node = 1:NumberOfLoads
    % This is created manually according to the naming of the folder
    FileDirectoryExtension= strcat('node_',num2str(node),'_pv_',num2str(TimeResolutionOfData),'_minute.mat');
    % The total file directory
    FileName=strcat(FileDirectoryBase,FileDirectoryExtension);
    % Load the 
    MatFile = load(FileName,'nodedata');    
    QSTS_Data(:,:,LoadList(node)) = MatFile.nodedata; %Putting the loads to appropriate nodes according to the loadlist
end

%% Seperate PV Generation Data
Generation = QSTS_Data(:,2,:);
Load = QSTS_Data(:,4,:);

% The above three lines still return a 3d matrix, where the column =1, so
% squeeze them
Generation=squeeze(Generation)*100/Sbase; % To convert to per unit, it should not be multiplied by 100
Load=squeeze(Load)/Sbase*100; % To convert to per unit
MaxGenerationPossible = max(Generation); % Getting the Maximum possible Generation for each Load 

%% Voltage Observer Parameters and related variable initialization
LowPassFilterFrequency = 0.1; % Low pass filter
HighPassFilterFrequency = 1; % high pass filter
Gain_Energy = 1e5; % gain Value
TimeStep=1;
% nc refers to no control, vqvp refers to Volt Var VoltWatt control 
FilteredOutput_nc = zeros(TotalTimeSteps,NumberOfNodes);
IntermediateOutput_nc=zeros(TotalTimeSteps,NumberOfNodes);
Epsilon_nc = zeros(TotalTimeSteps,NumberOfNodes);

FilteredOutput_vqvp = zeros(TotalTimeSteps,NumberOfNodes);
IntermediateOutput_vqvp=zeros(TotalTimeSteps,NumberOfNodes);
Epsilon_vqvp = zeros(TotalTimeSteps,NumberOfNodes);

%% ZIP load modeling
ConstantImpedanceFraction = 0.2; %constant impedance fraction of the load
ConstantCurrentFraction = 0.05; %constant current fraction of the load
ConstantPowerFraction = 0.75;  %constant power fraction of the load
ZIP_demand = zeros(TotalTimeSteps,IeeeFeeder,3); 

for node = 2:IeeeFeeder
    ZIP_demand(:,node,:) = [ConstantPowerFraction*Load(:,node), ConstantCurrentFraction*Load(:,node), ...
    ConstantImpedanceFraction*Load(:,node)]*(1 + 1i*tan(acos(power_factor))); % Q = P tan(theta)
end

%%  Power Flow For No Control Case - OpenDSS integration
SolarGeneration_NC= Generation * GenerationScalingFactor;

POpenDSS = zeros(TotalTimeSteps,length(LineNames));
QOpenDSS = POpenDSS;
VBusOpenDSS = zeros(TotalTimeSteps,length(BusNames));
VLoadOpenDSS = zeros(TotalTimeSteps,length(LoadBusNames));

[DSSObj, DSSText, gridpvpath] = DSSStartup;
% Load the components related to OpenDSS
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSMon=DSSCircuit.Monitors;

%Compile the Model
DSSText.command = 'Compile C:\feeders\feeder13_B_R\feeder13BR.dss';
% If all Load Parameters are to be changed
% DSSText.command ='batchedit load..* ZIPV= (0.2,0.05,0.75,0.2,0.05,0.75,0.8)';
% Get the load buses, according to the model to compare the simulation

% Set Slack Voltage = 1
setSourceInfo(DSSObj,{'source'},'pu',SlackBusVoltage);
for ksim=1:TotalTimeSteps
%     Change the real and reactive power of loads
    setLoadInfo(DSSObj,LoadBusNames,'kw',(Load(ksim,LoadList)-IncludeSolar*SolarGeneration_NC(ksim,LoadList))/1000); % To convert to KW
    setLoadInfo(DSSObj,LoadBusNames,'kvar',tan(acos(power_factor))*(Load(ksim,LoadList)-IncludeSolar*SolarGeneration_NC(ksim,LoadList))/1000);
    DSSSolution.Solve();
    
    %Get and store line info and bus info at each timestep
    LineInfo = getLineInfo(DSSObj,LineNames);
    QOpenDSS(ksim,:) = [LineInfo.bus1PowerReactive]';
    POpenDSS(ksim,:) = [LineInfo.bus1PowerReal]';
    
    BusInfo = getBusInfo(DSSObj,BusNames);
    VBusOpenDSS(ksim,:) = [BusInfo.voltagePU]';
    
    BusInfo = getBusInfo(DSSObj,LoadBusNames);
    VLoadOpenDSS(ksim,:) = [BusInfo.voltagePU]';
end

%% OpenDSS Modeling with Custom VQVP Control Case

%initalize - code to grab all of the line flows
POpenDSS_vqvp = zeros(TotalTimeSteps,length(LineNames));
QOpenDSS_vqvp = POpenDSS_vqvp;
VBusOpenDSS_vqvp = zeros(TotalTimeSteps,length(BusNames));
VLoadOpenDSS_vqvp = zeros(TotalTimeSteps,length(LoadBusNames));

SolarGeneration_vqvp= Generation * GenerationScalingFactor;
InverterReactivePower = zeros(TotalTimeSteps,length(LoadList));
InverterRealPower = zeros(TotalTimeSteps,length(LoadList));
InverterRateOfChangeLimit = 0.01; %rate of change limit
InverterRateOfChangeActivate = 0; %rate of change limit
% Droop Control Parameters
VQ_start = 1.01; VQ_end = 1.02; VP_start = 1.02; VP_end = 1.03;
% VQ_start = 1.5; VQ_end = 1.6; VP_start = 1.6; VP_end = 1.7;
VBP=ones(IeeeFeeder,1)*[VQ_start,VQ_end,VP_start,VP_end];
FilteredVoltage = zeros(TotalTimeSteps,length(LoadList));
InverterLPF = 1;
ThreshHold_vqvp=0.25;
% Setting up the maximum power capability
Sbar =  1.15 * MaxGenerationPossible * GenerationScalingFactor;

% Set Slack Voltage = 1
setSourceInfo(DSSObj,{'source'},'pu',SlackBusVoltage);
for ksim=1:TotalTimeSteps
%     Change the real and reactive power of loads
    if (ksim>1)
       setLoadInfo(DSSObj,LoadBusNames,'kw',(Load(ksim,LoadList)+IncludeSolar*InverterRealPower(ksim-1,:))/1000); % To convert to KW
       setLoadInfo(DSSObj,LoadBusNames,'kvar',tan(acos(power_factor))*(Load(ksim,LoadList)+IncludeSolar*InverterReactivePower(ksim-1,:))/1000); 
    else
       setLoadInfo(DSSObj,LoadBusNames,'kw',(Load(ksim,LoadList)-IncludeSolar*SolarGeneration_NC(ksim,LoadList))/1000); % To convert to KW
       setLoadInfo(DSSObj,LoadBusNames,'kvar',tan(acos(power_factor))*(Load(ksim,LoadList)-IncludeSolar*SolarGeneration_NC(ksim,LoadList))/1000); 
    end
    
    DSSSolution.Solve();
%    Get results
    LineInfo = getLineInfo(DSSObj,LineNames);    
    POpenDSS_vqvp(ksim,:) = [LineInfo.bus1PowerReal]';
    QOpenDSS_vqvp(ksim,:) = [LineInfo.bus1PowerReactive]';
    
    BusInfo = getBusInfo(DSSObj,BusNames);
    VBusOpenDSS_vqvp(ksim,:)=[BusInfo.voltagePU]';
    
    BusInfo = getBusInfo(DSSObj,LoadBusNames);
    VLoadOpenDSS_vqvp(ksim,:)=[BusInfo.voltagePU]';
    
    if(ksim > 1 && ksim < TotalTimeSteps)
        for node_iter = 1:NumberOfLoads
            knode = LoadList(node_iter);
            loadbusidx = LoadBusNames_ControlIndex(node_iter);
            [InverterReactivePower(ksim,node_iter),InverterRealPower(ksim,node_iter),FilteredVoltage(ksim,node_iter)] = ...
                inverter_VoltVarVoltWatt_model(FilteredVoltage(ksim-1,node_iter),...
                SolarGeneration_vqvp(ksim,knode),...
                VLoadOpenDSS_vqvp(ksim,loadbusidx),VLoadOpenDSS_vqvp(ksim-1,loadbusidx),...
                VBP(knode,:),TimeStep,InverterLPF,Sbar(knode),...
                InverterReactivePower(ksim-1,node_iter),InverterRealPower(ksim-1,node_iter),InverterRateOfChangeLimit,InverterRateOfChangeActivate);
        end
   end
end

%% Figures

t1 = datetime(2017,8,0,0,0,0);
t_datetime = t1 + minutes(Time);

NodeVoltageToPlot = find(contains(BusNames,'bus_632')==1);
NodePowerToPlot = find(contains(LineNames,'l_u_650')==1);

f1 = figure(1);
set(f1,'Units','Inches');
pos = get(f1,'Position');
set(f1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',...
    [ceil(pos(3)), ceil(pos(4))])
plot(t_datetime , VBusOpenDSS(:,NodeVoltageToPlot), 'b','LineWidth',1.5)
hold on
plot(t_datetime , VBusOpenDSS_vqvp(:,NodeVoltageToPlot), 'r','LineWidth',1.5)
hold off
legend('No Control','Control')
title('Voltage: bus 632')

f2 = figure(2);
set(f2,'Units','Inches');
pos = get(f2,'Position');
set(f2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime , POpenDSS(:,NodePowerToPlot), 'b','LineWidth',1.5)
hold on
plot(t_datetime , POpenDSS_vqvp(:,NodePowerToPlot), 'r','LineWidth',1.5)
hold off
legend('No Control','Control')
title('Real Power: line u to 650')
