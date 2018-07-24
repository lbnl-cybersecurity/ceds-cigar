% Author : Shammya Saha (sssaha@lbl.gov)
% Date : 05/22/2018
% This code shows the OpenDSS implementation of the in house built control technique and shows the comparison with the FBS 
% power flow tool developed.  

% UPDATED
% Author: Nate Tsang (nathan_tsang@lbl.gov)
% Date: 07/02/2018
% Now includes: 1) includes inverter control (originally created by Dan Arnold),
%               2) time delay (10-ish seconds),
%               3) time delay (2-ish minutes)
%               4) ability to change time steps from minutes to seconds
%               5) change time and number of hacked inverters for each node
%%
clc;
clear;
close all;

c = [];
q_avail = [];

%% Constants 
% These constants are used to create appropriate simulation scenario

LoadScalingFactor=2000;
GenerationScalingFactor=50;
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

%% Drawing Feeder
%feeder name for instance 13 means IEEE 13 Bus Test Feeder
IeeeFeeder=13;
% obtain Feeder map matrix - FM
% lengths between nodes - SL
% all paths from root to tail node - paths
% list of node names - nodelist
% Calling the Feeder mapper function
[FeederMap, Z_in_ohm, Paths, NodeList, LoadList] = ieee_feeder_mapper(IeeeFeeder);

%% Base Value Calculation

Vbase = 4.16e3; %4.16 kV
Sbase = 1; %500 kVA
Zbase = Vbase^2/Sbase;
Ibase = Sbase/Vbase;
Z = Z_in_ohm/Zbase; % Here Z represents Z in per unit. Transformer Not Included

%% Load Data Pertaining to Loads to create a profile
PV_Feeder_model=10; % resolution in minute
% Get the data from the Testpvnum folder
% Provide Your Directory
FileDirectoryBase='C:\Users\nathan_tsang\Desktop\LBL\CIGAR\ceds-cigar\LBNL_Simulations\testpvnum10\';

% Default simulation time
Time = 0:1440;

% Customize simulation time
StartTime = 60*9;
EndTime = 60*12;
Time = StartTime:EndTime;
TotalTimeSteps=length(Time);
QSTS_Data = zeros(length(Time),4,IeeeFeeder); % 4 columns as there are four columns of data available in the .mat file

for node = 1:NumberOfLoads
    % This is created manually according to the naming of the folder
    FileDirectoryExtension= strcat('node_',num2str(node),'_pv_',num2str(PV_Feeder_model),'_minute.mat');
    % The total file directory
    FileName=strcat(FileDirectoryBase,FileDirectoryExtension);
    % Load the 
    MatFile = load(FileName,'nodedata');    
    QSTS_Data(:,:,LoadList(node)) = MatFile.nodedata(StartTime:EndTime, :); %Putting the loads to appropriate nodes according to the loadlist
end

%% Seperate PV Generation Data
Generation = QSTS_Data(:,2,:)*GenerationScalingFactor;
Load = QSTS_Data(:,4,:)*LoadScalingFactor;

% The above three lines still return a 3d matrix, where the column =1, so
% squeeze them
Generation=squeeze(Generation)/Sbase; % To convert to per unit, it should not be multiplied by 100
Load=squeeze(Load)/Sbase; % To convert to per unit
MaxGenerationPossible = max(Generation); % Getting the Maximum possible Generation for each Load 

% %% Interpolate to change minutes to seconds and add noise
% 
% % Interpolate to get minutes to seconds
% for i = 1:NumberOfNodes
%     t_seconds = linspace(1,numel(Load(:,i)),3600*24/1);
%     LoadSeconds(:,i) = interp1(Load(:,i),t_seconds,'spline');
%     GenerationSeconds(:,i)= interp1(Generation(:,i),t_seconds,'spline');
% end 
% % Reassign variable names 
% TotalTimeSteps = 86400;
% Load = LoadSeconds;
% Generation = GenerationSeconds;
% % Create noise vector
% for node_iter = 1:NumberOfLoads
%     knode = LoadList(node_iter);
%         Noise(:,knode) = randn(TotalTimeSteps, 1);
% end 
% % Add noise to loads
% for i = 1:NumberOfNodes
%     Load(:,i) = LoadSeconds(:,i) + 2*Noise(:,i);
% end 
% 
% plot(Load(:,3))
% hold on
% plot(LoadSeconds(:,3))

%% Voltage Observer Parameters and related variable initialization
LowPassFilterFrequency = 0.1; % Low pass filter
HighPassFilterFrequency = 1; % high pass filter
Gain_Energy = 1e5; % gain Value
TimeStep=1;
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

%%  Power Flow with VQVP Control Case

% Initialize feeder states with control case
V_vqvp = zeros(IeeeFeeder,TotalTimeSteps);
S_vqvp = zeros(IeeeFeeder,TotalTimeSteps);
IterationCounter_vqvp = zeros(IeeeFeeder,TotalTimeSteps);
PowerEachTimeStep_vqvp = zeros(IeeeFeeder,3);

% Percentage of hacked inverters
SolarGeneration_vqvp_TOTAL = Generation * GenerationScalingFactor;
SolarGeneration_vqvp = SolarGeneration_vqvp_TOTAL;

TimeOfHack = 50;
PercentHacked = [0 0 1.0 0 0 0.5 0.5 0.5 0 0.5 0.5 0.5 0.5];
%% 
for t = TimeOfHack:TotalTimeSteps
    for i= 1:length(PercentHacked)
        SolarGeneration_vqvp(t,i) = SolarGeneration_vqvp_TOTAL(t,i) .* (1- PercentHacked(i));
        SolarGeneration_vqvp_hacked(t,i) = SolarGeneration_vqvp_TOTAL(t,i) .* PercentHacked(i);
    end 
end 

% Setting up the maximum power capability
Sbar =  MaxGenerationPossible * GenerationScalingFactor;
SbarHacked = Sbar .* PercentHacked;

%% 
InverterReactivePower = zeros(size(Generation));
InverterRealPower = zeros(size(Generation));
InverterReactivePowerHacked = zeros(size(Generation));
InverterRealPowerHacked = zeros(size(Generation));

InverterRateOfChangeLimit = 100; %rate of change limit
InverterRateOfChangeActivate = 0; %rate of change limit
% Droop Control Parameters
VQ_start = 1.01; VQ_end = 1.03; VP_start = 1.03; VP_end = 1.05;
VQ_startHacked = 1.01; VQ_endHacked = 1.015; VP_startHacked = 1.015; VP_endHacked = 1.02;

VBP = [nan*ones(IeeeFeeder,1,TotalTimeSteps), nan*ones(IeeeFeeder,1,TotalTimeSteps), ...
       nan*ones(IeeeFeeder,1,TotalTimeSteps), nan*ones(IeeeFeeder,1,TotalTimeSteps)];
VBPHacked = [nan*ones(IeeeFeeder,1,TotalTimeSteps), nan*ones(IeeeFeeder,1,TotalTimeSteps), ...
        nan*ones(IeeeFeeder,1,TotalTimeSteps), nan*ones(IeeeFeeder,1,TotalTimeSteps)];
   
VBP(:,1,1:2) = VQ_start;
VBP(:,2,1:2) = VQ_end;
VBP(:,3,1:2) = VP_start;
VBP(:,4,1:2) = VP_end;

VBPHacked(:,1,1:2) = VQ_startHacked;
VBPHacked(:,2,1:2) = VQ_endHacked;
VBPHacked(:,3,1:2) = VP_startHacked;
VBPHacked(:,4,1:2) = VP_endHacked;

FilteredVoltage = zeros(size(Generation));
FilteredVoltageCalc = zeros(size(Generation));
InverterLPF = 1;
ThreshHold_vqvp = 0.25;
V0=SlackBusVoltage*ones(TotalTimeSteps,1);

% Adaptive controller parameters
upk = zeros(size(IntermediateOutput_vqvp));
uqk = upk;
kq = 100;
kp = 100;

% Delays                [1 2 3* 4 5  6*  7*  8*  9*  10* 11* 12* 13*
Delay_VoltageSampling = [0 0 1 0 0 1 1 1 1 1 1 1 1]; 
Delay_VBPCurveShift =   [0 0 2 0 0 2 2 2 2 2 2 2 2]; 

%% SIMULATION

for ksim=1:TotalTimeSteps
   
   % CALCULATE NET ZIP LOADS
   for node_iter = 1:NumberOfLoads
        knode = LoadList(node_iter);
        if (ksim>1)
            PowerEachTimeStep_vqvp(knode,:) = [ZIP_demand(ksim,knode,1) + ...
                InverterRealPower(ksim-1,knode) + InverterRealPowerHacked(ksim-1,knode) + ...
                1i*InverterReactivePower(ksim-1,knode) + 1i*InverterReactivePowerHacked(ksim-1,knode), ...
                ZIP_demand(ksim,knode,2), ZIP_demand(ksim,knode,3)];
        else
            PowerEachTimeStep_vqvp(knode,:) = [ZIP_demand(ksim,knode,1) - ...
                SolarGeneration_vqvp(ksim,knode) - SolarGeneration_vqvp_hacked(ksim,knode), ...
                ZIP_demand(ksim,knode,2),ZIP_demand(ksim,knode,3)];
        end
   end
   
   % RUN FORWARD-BACKWARD SWEEP
   [V_vqvp(:,ksim),Irun,S_vqvp(:,ksim),IterationCounter_vqvp(:,ksim)] = ...
       FBSfun(V0(ksim),PowerEachTimeStep_vqvp,Z,FeederMap);
    
   % RUN INVERTER FUNCTION TO OUTPUT P/Q
   if(ksim > 1 && ksim < TotalTimeSteps)
        for node_iter = 1:NumberOfLoads
            knode = LoadList(node_iter);
            
            % Inverter (not hacked)
            [InverterReactivePower(ksim,knode),InverterRealPower(ksim,knode),...
            FilteredVoltage(ksim:TotalTimeSteps,knode), FilteredVoltageCalc(ksim,knode),...
            c(ksim,knode), q_avail(ksim,knode)] = ...
            inverter_model(FilteredVoltage(ksim-1,knode),...
            SolarGeneration_vqvp(ksim,knode),...
            abs(V_vqvp(knode,ksim)),abs(V_vqvp(knode,ksim-1)),...
            VBP(knode,:,ksim),TimeStep,InverterLPF,Sbar(knode),...
            InverterRealPower(ksim-1,knode),InverterReactivePower(ksim-1,knode),...
            InverterRateOfChangeLimit,InverterRateOfChangeActivate,...
            ksim,Delay_VoltageSampling(knode)); 
            
            % Inverter (hacked)
            [InverterReactivePowerHacked(ksim,knode),InverterRealPowerHacked(ksim,knode),...
            FilteredVoltage(ksim:TotalTimeSteps,knode), FilteredVoltageCalc(ksim,knode),...
            c(ksim,knode), q_avail(ksim,knode)] = ...
            inverter_model(FilteredVoltage(ksim-1,knode),...
            SolarGeneration_vqvp_hacked(ksim,knode),...
            abs(V_vqvp(knode,ksim)),abs(V_vqvp(knode,ksim-1)),...
            VBPHacked(knode,:,ksim),TimeStep,InverterLPF,SbarHacked(knode),...
            InverterRealPowerHacked(ksim-1,knode),InverterReactivePowerHacked(ksim-1,knode),...
            InverterRateOfChangeLimit,InverterRateOfChangeActivate,...
            ksim,Delay_VoltageSampling(knode));         
        end
   end
    
   % RUN OBSERVER FUNCTION
   for node_iter=1:NumberOfLoads %cycle through each node
        knode = LoadList(node_iter);
        if (ksim>1)
            [FilteredOutput_vqvp(ksim,knode),IntermediateOutput_vqvp(ksim,knode), ... 
                Epsilon_vqvp(ksim,knode)] = voltage_observer(V_vqvp(knode,ksim), ...
                V_vqvp(knode,ksim-1), IntermediateOutput_vqvp(ksim-1,knode), ...
                Epsilon_vqvp(ksim-1,knode), FilteredOutput_vqvp(ksim-1,knode), ... 
                HighPassFilterFrequency, LowPassFilterFrequency, Gain_Energy, TimeStep);
        end
        
   % RUN ADAPTIVE CONTROLLER
        if mod(ksim, Delay_VBPCurveShift(knode)) == 0 
            %re-set VBP
            upk(ksim,knode) = adaptive_control(Delay_VBPCurveShift(knode), ...
                kp, IntermediateOutput_vqvp(ksim,knode), ...
                IntermediateOutput_vqvp(ksim-Delay_VBPCurveShift(knode)+1,knode), ...
                upk(ksim-Delay_VBPCurveShift(knode)+1,knode), ThreshHold_vqvp, ...
                FilteredOutput_vqvp(ksim,knode));

            uqk(ksim,knode) = adaptive_control(Delay_VBPCurveShift(knode), ...
                kq, IntermediateOutput_vqvp(ksim,knode), ...
                IntermediateOutput_vqvp(ksim-Delay_VBPCurveShift(knode)+1,knode), ....
                uqk(ksim-Delay_VBPCurveShift(knode)+1,knode), ThreshHold_vqvp, ...
                FilteredOutput_vqvp(ksim,knode));
     
   % CALCULATE NEW VBP1
            for j = ksim:TotalTimeSteps
                VBP(knode,:,j) = [VQ_start - uqk(ksim,knode),...
                 VQ_end + uqk(ksim,knode), VP_start - upk(ksim,knode), VP_end + upk(ksim,knode)];  
             
                upk(j,knode) = upk(ksim,knode);
                uqk(j,knode) = uqk(ksim,knode);
                
            end 
        else
            if ksim > 1
                for j = ksim:TotalTimeSteps
                    VBP(knode,:,j) = VBP(knode,:,j-1);
                end 
            end 
        end
   % CALCULATE NEW VBP_HACKED
        if ksim > 1
            for j = ksim:TotalTimeSteps
                VBPHacked(knode,:,j) = VBPHacked(knode,:,j-1); 
            end 
        end 
   end
end


P0_vqvp = real(S_vqvp(1,:));
Q0_vqvp = imag(S_vqvp(1,:));

% %% OpenDSS Modeling with Custom VQVP Control Case
% 
% [DSSObj, DSSText, gridpvpath] = DSSStartup;
% % Load the components related to OpenDSS
% DSSCircuit=DSSObj.ActiveCircuit;
% DSSSolution=DSSCircuit.Solution;
% DSSMon=DSSCircuit.Monitors;
% 
% %Compile the Model
% DSSText.command = 'Compile C:\feeders\feeder13_B_R\feeder13BR.dss';
% % If all Load Parameters are to be changed
% % DSSText.command ='batchedit load..* ZIPV= (0.2,0.05,0.75,0.2,0.05,0.75,0.8)';
% % Get the load buses, according to the model to compare the simulation
% LoadBus=NodeList(LoadList);
% LoadBusNames=cell(1,8);
% for i = 1:length(LoadBus)
%     LoadBusNames{i} = strcat('load_',num2str(LoadBus(i)));
% end
NodeVoltageToPlot=680;
% 
% InverterReactivePower = zeros(TotalTimeSteps,length(LoadList));
% InverterRealPower = zeros(TotalTimeSteps,length(LoadList));
% FilteredVoltage = zeros(TotalTimeSteps,length(LoadList));
% VoltageOpenDSS_vqvp=zeros(length(LoadList),TotalTimeSteps);
% Voltage_Bus680=zeros(1,TotalTimeSteps);
% SubstationRealPowerOpenDSS_vqvp=zeros(1,TotalTimeSteps);
% SubstationReactivePowerOpenDSS_vqvp=zeros(1,TotalTimeSteps);
% % Set Slack Voltage = 1
% setSourceInfo(DSSObj,{'source'},'pu',SlackBusVoltage);
% for ksim=1:TotalTimeSteps
% %     Change the real and reactive power of loads
%     if (ksim>1)
%        setLoadInfo(DSSObj,LoadBusNames,'kw',(Load(ksim,LoadList)+IncludeSolar*InverterRealPower(ksim-1,:))/1000); % To convert to KW
%        setLoadInfo(DSSObj,LoadBusNames,'kvar',tan(acos(power_factor))*(Load(ksim,LoadList)+IncludeSolar*InverterReactivePower(ksim-1,:))/1000); 
%     else
%        setLoadInfo(DSSObj,LoadBusNames,'kw',(Load(ksim,LoadList)-IncludeSolar*SolarGeneration_vqvp(ksim,LoadList))/1000); % To convert to KW
%        setLoadInfo(DSSObj,LoadBusNames,'kvar',tan(acos(power_factor))*(Load(ksim,LoadList)-IncludeSolar*SolarGeneration_vqvp(ksim,LoadList))/1000); 
%     end
%     
%     DSSSolution.Solve();
% %    Get results
%     LineInfo = getLineInfo(DSSObj,LineNames);    
%     POpenDSS_vqvp(ksim,:) = [LineInfo.bus1PowerReal]';
%     QOpenDSS_vqvp(ksim,:) = [LineInfo.bus1PowerReactive]';
%     
%     BusInfo = getBusInfo(DSSObj,BusNames);
%     VBusOpenDSS_vqvp(ksim,:)=[BusInfo.voltagePU]';
%     
%     BusInfo = getBusInfo(DSSObj,LoadBusNames);
%     VLoadOpenDSS_vqvp(ksim,:)=[BusInfo.voltagePU]';
%     
%     if(ksim > 1 && ksim < TotalTimeSteps)
%         for node_iter = 1:NumberOfLoads
%             knode = LoadList(node_iter);
%             [InverterReactivePower(ksim,node_iter),InverterRealPower(ksim,node_iter),FilteredVoltage(ksim,node_iter)] = ...
%                 inverter_VoltVarVoltWatt_model(FilteredVoltage(ksim-1,node_iter),...
%                 SolarGeneration_vqvp(ksim,knode),...
%                 VoltageOpenDSS_vqvp(node_iter,ksim),VoltageOpenDSS_vqvp(node_iter,ksim-1),...
%                 VBP(knode,:),TimeStep,InverterLPF,Sbar(knode),...
%                 InverterReactivePower(ksim-1,node_iter),InverterRealPower(ksim-1,node_iter),...
%                 InverterRateOfChangeLimit,InverterRateOfChangeActivate,ksim,Delay_VoltageSampling);
%         end
%    end
% end

%% Figures
t1 = datetime(2017,8,0,0,0,0);
t_datetime = t1 + minutes(Time);
t_hacktime = t1 + minutes(TimeOfHack+StartTime);

%
node = 8;
f1 = figure(1);
subplot(2,1,1)
hold on
plot(t_datetime , P0_vqvp, 'b','LineWidth',1.5)
plot(t_hacktime, line([t_hacktime t_hacktime], get(gca, 'ylim')), 'LineWidth', 1.5)
hold off
title('Real Power (W) From Substation')
subplot(2,1,2)
hold on
plot(t_datetime , Q0_vqvp, 'r','LineWidth',1.5)
plot(t_hacktime, line([t_hacktime t_hacktime], get(gca, 'ylim')), 'LineWidth', 1.5)
hold off
title('Reactive Power (VA) From Substation')

f2 = figure(2);
subplot(2,1,1)
plot(t_datetime , squeeze(VBP(node,1,:)), t_datetime, squeeze(VBP(node,2,:)), ...
    t_datetime, squeeze(VBP(node,3,:)), t_datetime, squeeze(VBP(node,4,:)), ...
    'LineWidth',1.5)
title('VBP for Node 3')
legend({'V1', 'V2', 'V3', 'V4'},'FontSize',12)
subplot(2,1,2)
plot(t_datetime , squeeze(VBPHacked(node,1,:)), t_datetime, squeeze(VBPHacked(node,2,:)), ...
    t_datetime, squeeze(VBPHacked(node,3,:)), t_datetime, squeeze(VBPHacked(node,4,:)), ...
    'LineWidth',1.5)
title('VBPHacked for Node 3')
legend({'V1', 'V2', 'V3', 'V4'},'FontSize',12)
ylim([1.005 1.035])

f3 = figure(3);
plot(t_datetime , abs(V_vqvp(node,:)), 'r','LineWidth',1.5)
datetick('x','HH:MM')
title(['Voltage Magnitude, node: 3'])
xlabel('time (hours)')
ylabel('volts (pu)')

f4 = figure(4);
hold on
plot(t_datetime, FilteredVoltage(:,node), 'b','LineWidth',1.5 )
plot(t_datetime, FilteredVoltageCalc(:,node), 'r','LineWidth',1.5 )
hold off
datetick('x','HH:MM')
title(['Inverter: LP Filter of voltage magnitude from inverter, node: 3'])
xlabel('time (hours)')
ylabel('V?')
legend({'Filtered voltage used','Filtered voltage calculated'},'FontSize',12);

f5 = figure(5);
subplot(2,1,1)
plot(t_datetime, upk(:,node), t_datetime, uqk(:,node),'LineWidth',1.5 )
title(['Adaptive controller output, node: 3'])
legend({'upk','uqk'},'FontSize',12);

subplot(2,1,2)
hold on
plot(t_datetime, IntermediateOutput_vqvp(:,node), 'r','LineWidth',1.5 )
plot(t_datetime, FilteredOutput_vqvp(:,node), 'y','LineWidth',1.5 )
plot(t_datetime, ThreshHold_vqvp*ones(1,TotalTimeSteps), 'm','LineWidth',1.5 )
plot(t_hacktime, line([t_hacktime t_hacktime], get(gca, 'ylim')), 'LineWidth', 1.5)
hold off 
datetick('x','HH:MM')
title(['Adaptive controller input, node: 3'])
xlabel('time (hours)')
ylabel('V?')
ylim([0 0.26])
legend({'Intermediate output', 'filtered output', 'threshold'},'FontSize',12);

% Voltage observer inputs and outputs
f6 = figure(6);
plot(t_datetime, FilteredOutput_vqvp(:,node), 'LineWidth',1.5 )
legend({'filtered output'},'FontSize',12);
title(['Observer outputs'])

% f6 = figure(6);
% % subplot(2,1,1)
% plot(t_datetime, FilteredOutput_vqvp(:,node), t_datetime, IntermediateOutput_vqvp(:,node),...
%     t_datetime, Epsilon_vqvp(:,node), 'LineWidth',1.5 )
% legend({'filtered output', 'intermediate output', 'epsilon'},'FontSize',12);
% title(['Observer outputs'])

% subplot(2,1,2)
% plot(t_datetime, abs(V_vqvp(node,:)), 'r','LineWidth',1.5 )
% datetick('x','HH:MM')
% title(['Observer inputs'])
% xlabel('time (hours)')
% ylabel('V?')
% % ylim([0 0.26])
% legend({'voltage'},'FontSize',12);

f7 = figure(7);
subplot(2,1,1)
hold on
plot(t_datetime, InverterReactivePower(:,node), t_datetime, InverterReactivePowerHacked(:,node), 'LineWidth',1.5) 
plot(t_hacktime, line([t_hacktime t_hacktime], get(gca, 'ylim')), 'r', 'LineWidth', 1.5)
hold off
title(['Inverter reactive power - Node 3'])
legend({'inverter reactive power- not hacked', 'inverter reactive power - hacked', 'time of hack'},'FontSize',12);

subplot(2,1,2)
hold on
plot(t_datetime, InverterRealPower(:,node), t_datetime, InverterRealPowerHacked(:,node), 'LineWidth',1.5)
plot(t_hacktime, line([t_hacktime t_hacktime], get(gca, 'ylim')), 'r', 'LineWidth', 1.5)
hold off
title(['Inverter real power - Node 3'])
legend({'inverter real power- not hacked', 'inverter real power - hacked', 'time of hack'},'FontSize',12);


f8 = figure(8);
plot(t_datetime, sum(Load'), 'r', 'LineWidth',1.5)
hold on
plot(t_hacktime, line([t_hacktime t_hacktime], get(gca, 'ylim')), 'r', 'LineWidth', 1.5)
datetick('x','HH:MM')
title(['Sum of Loads'])
xlabel('time (hours)')
ylabel('VA')
legend('loads', 'time of hack')

f9 = figure(9);
plot(t_datetime, InverterRealPower(:,3) + InverterRealPowerHacked(:,3), ...
    t_datetime, InverterRealPower(:,6) + InverterRealPowerHacked(:,6), ...
    t_datetime, InverterRealPower(:,7) + InverterRealPowerHacked(:,7), ...
    t_datetime, InverterRealPower(:,8) + InverterRealPowerHacked(:,8), ...
    t_datetime, InverterRealPower(:,10) + InverterRealPowerHacked(:,10), ...
    t_datetime,InverterRealPower(:,11) + InverterRealPowerHacked(:,11), ...
    t_datetime,InverterRealPower(:,12) + InverterRealPowerHacked(:,12),...
    t_datetime, InverterRealPower(:,13) + InverterRealPowerHacked(:,13), 'LineWidth',1.5)
title(['Inverter real power output for each load - combined (hacked+not hacked)'])
legend({'Node 3', 'Node 6', 'Node 7', 'Node 8', 'Node 10', 'Node 11', 'Node 12', 'Node 13'},'FontSize',12);
