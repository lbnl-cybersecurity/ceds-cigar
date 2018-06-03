% Author : Shammya Saha (sssaha@lbl.gov)
% Date : 05/22/2018
% This code shows the OpenDSS implementation of the in house built control technique and shows the comparison with the FBS 
% power flow tool developed.  

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
%% Drawing Feeder
%feeder name for instance 13 means IEEE 13 Bus Test Feeder
IeeeFeeder=13;
% obtain Feeder map matrix - FM
% lengths between nodes - SL
% all paths from root to tail node - paths
% list of node names - nodelist
% Calling the Feeder mapper function
[FeederMap, Z_in_ohm, Paths, NodeList, LoadList] = ieee_feeder_mapper(IeeeFeeder);

NumberOfLoads=length(LoadList);
NumberOfNodes=length(NodeList);

% setup voltages and currents
% Vbase = 4.16e3; %4.16 kV
% Sbase = 500e3; %500 kVA

%% Base Value Calculation

Vbase = 4.16e3; %4.16 kV
Sbase = 1; %500 kVA

Zbase = Vbase^2/Sbase;
Ibase = Sbase/Vbase;

Z = Z_in_ohm/Zbase; % Here Z represents Z in per unit. Transformer Not Included

%% Load Data Pertaining to Loads to create a profile
TimeResolutionOfData=10; % resolution in minute
% Get the data from the Testpvnum folder
% Provide Your Directory
FileDirectoryBase='C:\Users\shamm\Dropbox (ASU)\ASU\Microgrid Controller\CIGAR\CEDS_CIGAR\LBNL_Simulations\testpvnum10\';
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



%%  Power Flow For No Control Case
V0=SlackBusVoltage*ones(TotalTimeSteps,1);
%Initialize feeder states, nc means no control present
V_nc = zeros(IeeeFeeder,TotalTimeSteps);
S_nc =  zeros(IeeeFeeder,TotalTimeSteps);
IterationCounter_nc=zeros(IeeeFeeder,TotalTimeSteps);
PowerEachTimeStep_nc = zeros(IeeeFeeder,3);
% Setting up the maximum power capability
Sbar =  1.15 * MaxGenerationPossible * GenerationScalingFactor;
% Increasing the generation capability
SolarGeneration_NC= Generation * GenerationScalingFactor;
% for ksim=1:TotalTimeSteps
for ksim=1:TotalTimeSteps    
   for node_iter = 1:NumberOfLoads
       knode = LoadList(node_iter);
       % Checking whether manipulation had made the code wrong
       if(SolarGeneration_NC(ksim,knode) > Sbar(knode))
          SolarGeneration_NC(ksim,knode) = Sbar(knode);
       end
   % Doing load - generation provides us the net load at each time step    
   PowerEachTimeStep_nc(knode,:) = [ZIP_demand(ksim,knode,1) - IncludeSolar*SolarGeneration_NC(ksim,knode),...
            ZIP_demand(ksim,knode,2),ZIP_demand(ksim,knode,3)];    
   end
   % Doing the FBS power flow
   [V_nc(:,ksim),Irun,S_nc(:,ksim),IterationCounter_nc(:,ksim)] = FBSfun(V0(ksim),PowerEachTimeStep_nc,Z,FeederMap);
   
end
P0_nc = real(S_nc(1,:));
Q0_nc = imag(S_nc(1,:));
 

%% OpenDSS integration

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
LoadBus=NodeList(LoadList);
LoadBusNames=cell(1,8);
for i = 1:length(LoadBus)
    LoadBusNames{i} = strcat('load_',num2str(LoadBus(i)));
end
NodeVoltageToPlot=680;
VoltageOpenDSS=zeros(1,TotalTimeSteps);
SubstationRealPowerOpenDSS=zeros(1,TotalTimeSteps);
% Set Slack Voltage = 1
setSourceInfo(DSSObj,{'source'},'pu',SlackBusVoltage);
for ksim=1:TotalTimeSteps
%     Change the real and reactive power of loads
    setLoadInfo(DSSObj,LoadBusNames,'kw',(Load(ksim,LoadList)-IncludeSolar*SolarGeneration_NC(ksim,LoadList))/1000); % To convert to KW
    setLoadInfo(DSSObj,LoadBusNames,'kvar',tan(acos(power_factor))*(Load(ksim,LoadList)-IncludeSolar*SolarGeneration_NC(ksim,LoadList))/1000);
    DSSSolution.Solve();
%     Getting the incoming power flow
    LineInfo = getLineInfo(DSSObj, {'L_U_650'});
    SubstationRealPowerOpenDSS(1,ksim)=LineInfo.bus1PowerReal;
%     GEt the Bus 634 Voltage
    BusInfo=getBusInfo(DSSObj,{strcat('bus_',num2str(NodeVoltageToPlot))});
    VoltageOpenDSS(1,ksim)=BusInfo.voltagePU;
end

%% Figures - overlay

t1 = datetime(2017,8,0,0,0,0);
t_datetime = t1 + minutes(Time);

node = find(NodeList==NodeVoltageToPlot);
f1 = figure(1);
set(f1,'Units','Inches');
pos = get(f1,'Position');
set(f1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime , abs(V_nc(node,:)), 'b','LineWidth',1.5)
hold on;
plot(t_datetime , VoltageOpenDSS, 'r','LineWidth',1.5)
legend('FBS','OpenDSS')

f2 = figure(2);
set(f2,'Units','Inches');
pos = get(f2,'Position');
set(f2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime , P0_nc, 'b','LineWidth',1.5)
hold on;
plot(t_datetime , 1000*SubstationRealPowerOpenDSS, 'r','LineWidth',1.5) % TO convert to Watts
legend('FBS','OpenDSS')

% 
%% %%  Power Flow with VQVP Control Case
% Initialize feeder states with control case
V_vqvp =  zeros(IeeeFeeder,TotalTimeSteps);
S_vqvp =  zeros(IeeeFeeder,TotalTimeSteps);
IterationCounter_vqvp=zeros(IeeeFeeder,TotalTimeSteps);
PowerEachTimeStep_vqvp = zeros(IeeeFeeder,3);
SolarGeneration_vqvp= Generation * GenerationScalingFactor;
InverterReactivePower = zeros(size(Generation));
InverterRealPower = zeros(size(Generation));
InverterRateOfChangeLimit = 0.01; %rate of change limit
InverterRateOfChangeActivate = 0; %rate of change limit
% Droop Control Parameters
VQ_start = 1.01; VQ_end = 1.02; VP_start = 1.02; VP_end = 1.03;
% VQ_start = 1.5; VQ_end = 1.6; VP_start = 1.6; VP_end = 1.7;
VBP=ones(IeeeFeeder,1)*[VQ_start,VQ_end,VP_start,VP_end];

FilteredVoltage = zeros(size(Generation));

InverterLPF = 1;
ThreshHold_vqvp=0.25;
V0=SlackBusVoltage*ones(TotalTimeSteps,1);
for ksim=1:TotalTimeSteps
   for node_iter = 1:NumberOfLoads
        knode = LoadList(node_iter);
        if (ksim>1)
            PowerEachTimeStep_vqvp(knode,:) = [ZIP_demand(ksim,knode,1) + InverterRealPower(ksim-1,knode) + ...
                1i*InverterReactivePower(ksim-1,knode), ZIP_demand(ksim,knode,2), ...
                ZIP_demand(ksim,knode,3)];
        else
            PowerEachTimeStep_vqvp(knode,:) = [ZIP_demand(ksim,knode,1) - SolarGeneration_vqvp(ksim,knode), ...
                ZIP_demand(ksim,knode,2),ZIP_demand(ksim,knode,3)];
        end
   end
   [V_vqvp(:,ksim),Irun,S_vqvp(:,ksim),IterationCounter_vqvp(:,ksim)] = FBSfun(V0(ksim),PowerEachTimeStep_vqvp,Z,FeederMap);

   if(ksim > 1 && ksim < TotalTimeSteps)
        for node_iter = 1:NumberOfLoads
            knode = LoadList(node_iter);
            [InverterReactivePower(ksim,knode),InverterRealPower(ksim,knode),FilteredVoltage(ksim,knode)] = ...
                inverter_VoltVarVoltWatt_model(FilteredVoltage(ksim-1,knode),...
                SolarGeneration_vqvp(ksim,knode),...
                abs(V_vqvp(knode,ksim)),abs(V_vqvp(knode,ksim-1)),...
                VBP(knode,:),TimeStep,InverterLPF,Sbar(knode),...
                InverterReactivePower(ksim-1,knode),InverterRealPower(ksim-1,knode),InverterRateOfChangeLimit,InverterRateOfChangeActivate);
        end
   end

end

P0_vqvp = real(S_vqvp(1,:));
Q0_vqvp = imag(S_vqvp(1,:));



%% OpenDSS Modeling with Custom VQVP Control Case

InverterReactivePower = zeros(TotalTimeSteps,length(LoadList));
InverterRealPower = zeros(TotalTimeSteps,length(LoadList));
FilteredVoltage = zeros(TotalTimeSteps,length(LoadList));
VoltageOpenDSS_vqvp=zeros(length(LoadList),TotalTimeSteps);
Voltage_Bus680=zeros(1,TotalTimeSteps);
SubstationRealPowerOpenDSS_vqvp=zeros(1,TotalTimeSteps);
SubstationReactivePowerOpenDSS_vqvp=zeros(1,TotalTimeSteps);
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
%    Getting the incoming power flow
    LineInfo = getLineInfo(DSSObj, {'L_U_650'});
    SubstationRealPowerOpenDSS_vqvp(1,ksim)=LineInfo.bus1PowerReal;
    SubstationReactivePowerOpenDSS_vqvp(1,ksim)=LineInfo.bus1PowerReactive;
%     GEt the Bus 634 Voltage
    BusInfo=getBusInfo(DSSObj,LoadBusNames);
    VoltageOpenDSS_vqvp(:,ksim)=[BusInfo.voltagePU]';
    BusInfo=getBusInfo(DSSObj,strcat('bus_',num2str(NodeVoltageToPlot)));
    Voltage_Bus680(1,ksim)=BusInfo.voltagePU;
    if(ksim > 1 && ksim < TotalTimeSteps)
        for node_iter = 1:NumberOfLoads
            knode = LoadList(node_iter);
            [InverterReactivePower(ksim,node_iter),InverterRealPower(ksim,node_iter),FilteredVoltage(ksim,node_iter)] = ...
                inverter_VoltVarVoltWatt_model(FilteredVoltage(ksim-1,node_iter),...
                SolarGeneration_vqvp(ksim,knode),...
                VoltageOpenDSS_vqvp(node_iter,ksim),VoltageOpenDSS_vqvp(node_iter,ksim-1),...
                VBP(knode,:),TimeStep,InverterLPF,Sbar(knode),...
                InverterReactivePower(ksim-1,node_iter),InverterRealPower(ksim-1,node_iter),InverterRateOfChangeLimit,InverterRateOfChangeActivate);
        end
   end
end

%% Figures
t1 = datetime(2017,8,0,0,0,0);
t_datetime = t1 + minutes(Time);

node = find(NodeList==NodeVoltageToPlot);
f1 = figure(3);
set(f1,'Units','Inches');
pos = get(f1,'Position');
set(f1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime , abs(V_vqvp(node,:)), 'b','LineWidth',1.5)
hold on;
plot(t_datetime , Voltage_Bus680(1,:), 'r','LineWidth',1.5)
legend('VQVP','OpenDSS')
%%
f2 = figure(4);
set(f2,'Units','Inches');
pos = get(f2,'Position');
set(f2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
subplot(211)
plot(t_datetime , P0_vqvp/1000, 'b','LineWidth',1.5)
hold on;
plot(t_datetime , SubstationRealPowerOpenDSS_vqvp, 'r','LineWidth',1.5) % TO convert to Watts
legend('FBS','OpenDSS')
title('Real Power (kW) From Substation')

subplot(212)
plot(t_datetime , Q0_vqvp/1000, 'b','LineWidth',1.5)
hold on;
plot(t_datetime , SubstationReactivePowerOpenDSS_vqvp, 'r','LineWidth',1.5) % TO convert to Watts
legend('FBS','OpenDSS')
title('Reactive Power (kVAR) From Substation')
