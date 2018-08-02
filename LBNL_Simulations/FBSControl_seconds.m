%{ 
Project : GMLC 1.4.23 - Inverter Attack Analysis
Authors : Nate Tsang (nathan_tsang@lbl.gov), Daniel Arnold (dbarnold@lbl.gov), and Shammya Saha (sssaha@lbl.gov)
Last Update : 08/3/2018

Purpose: This code simulates 1) how a cyber-physical attack can cause voltage
instability and 2) how smart inverter controls can regain stability by only
sensing voltage at the local node. This utilizes the IEEE 13 BUS FEEDER and
the simulation runs at a SECOND timescale.

The following code has the following functionality:
1) Ability to change simulation start/stop time
2) Run power flow using FBS using 'FBSfun.m'
3) Simulate inverter P/Q contributions using 'inverter_model.m'
4) Include delays for voltage sampling (~10 sec) and VBP parameter adjustment
   (~120 sec), which can be customized for each node
5) Utilize observer to filter voltage for determining the energy of
   instabilities using 'voltage_observer.m'
6) Include adaptive controller using 'adaptive_control.m'
7) Ability to simulate cyber-physical attack at a time step determined by
   the user
8) Ability to change which nodes are hacked, and what percentage of
   inverters at that node are hacked.
9) Display figures

%}

clc;
clear;
close all;

%% TUNING KNOBS - ADJUST (MAIN) PARAMETERS HERE 
% Note that there are other tunable parameters in the code, but these are
% the main ones

% Set load factors and slack voltage
LoadScalingFactor = 2000; 
GenerationScalingFactor = 70; 
SlackBusVoltage = 1.02; 

% Set simulation analysis period
StartTime = 40000; 
EndTime = 40500; 

% Set hack parameters
TimeStepOfHack = 50;
PercentHacked = [0 0 1.0 0 0 0.5 0.5 0.5 0 0.5 0.5 0.5 0.5];

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

%% Feeder parameters
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

%% Draw Feeder 
%feeder name for instance 13 means IEEE 13 Bus Test Feeder
IeeeFeeder=13;
% obtain Feeder map matrix - FM
% lengths between nodes - SL
% all paths from root to tail node - paths
% list of node names - nodelist
% Calling the Feeder mapper function
[FeederMap, Z_in_ohm, Paths, NodeList, LoadList] = ieee_feeder_mapper(IeeeFeeder);

% Base Value Calculations
Vbase = 4.16e3; %4.16 kV
Sbase = 1; % Set to something like '500 kVA' to use per unit
Zbase = Vbase^2/Sbase;
Ibase = Sbase/Vbase;
Z = Z_in_ohm/Zbase; % Here Z represents Z in per unit. Transformer Not Included

power_factor=0.9;
IncludeSolar=1; % Setting this to 1 will include the solar , set to zero if no solar required

% Load Data Pertaining to Loads to create a profile
PV_Feeder_model=10; 

% Get the data from the Testpvnum folder
FileDirectoryBase='C:\Users\nathan_tsang\Desktop\LBL\CIGAR\ceds-cigar\LBNL_Simulations\testpvnum10\';

QSTS_Time = 0:1440; % Default data points in load data file (GSTS_Data) - don't change
QSTS_Data = zeros(length(QSTS_Time),4,IeeeFeeder); % Retrieve load data

for node = 1:NumberOfLoads
    % This is created manually according to the naming of the folder
    FileDirectoryExtension= strcat('node_',num2str(node),'_pv_',num2str(PV_Feeder_model),'_minute.mat');
    % The total file directory
    FileName=strcat(FileDirectoryBase,FileDirectoryExtension);
    
    MatFile = load(FileName,'nodedata');    
    QSTS_Data(:,:,LoadList(node)) = MatFile.nodedata;
end

%% Seperate PV Generation Data
Generation = QSTS_Data(:,2,:)*GenerationScalingFactor; % solar generation
Load = QSTS_Data(:,4,:)*LoadScalingFactor; % load demand

% The above three lines still return a 3d matrix, where the column =1, so
% squeeze them into a 2D matrix
Generation=squeeze(Generation)/Sbase; % To convert to per unit, it should not be multiplied by 100
Load=squeeze(Load)/Sbase; % To convert to per unit
MaxGenerationPossible = max(Generation); % Getting the Maximum possible Generation for each Load 

%% Interpolate to change data from minutes to seconds
Time = StartTime:EndTime;
TotalTimeSteps=length(Time);

% Interpolate to get minutes to seconds
for i = 1:NumberOfNodes
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
for node_iter = 1:NumberOfLoads
    knode = LoadList(node_iter);
        Noise(:,knode) = randn(TotalTimeSteps, 1);
end 

% Add noise to loads
for i = 1:NumberOfNodes
    Load(:,i) = LoadSeconds(:,i) + 2*Noise(:,i);
end 

%% Voltage Observer Parameters and related variable initialization
LowPassFilterFrequency = 0.1; % Low pass filter
HighPassFilterFrequency = 1; % high pass filter
Gain_Energy = 1e5;
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

%%  Power Flow with VQVP Control
% Initialize feeder states with control case
V_vqvp = zeros(IeeeFeeder,TotalTimeSteps);
S_vqvp = zeros(IeeeFeeder,TotalTimeSteps);
IterationCounter_vqvp = zeros(IeeeFeeder,TotalTimeSteps);
PowerEachTimeStep_vqvp = zeros(IeeeFeeder,3);

% Percentage of irradiation "seen by" uncompromised and hacked inverters
SolarGeneration_vqvp_TOTAL = Generation * GenerationScalingFactor;
SolarGeneration_vqvp = SolarGeneration_vqvp_TOTAL;

for t = TimeStepOfHack:TotalTimeSteps
    for i= 1:length(PercentHacked)
        SolarGeneration_vqvp(t,i) = SolarGeneration_vqvp_TOTAL(t,i) .* (1- PercentHacked(i));
        SolarGeneration_vqvp_hacked(t,i) = SolarGeneration_vqvp_TOTAL(t,i) .* PercentHacked(i);
    end 
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

% Initialization
InverterReactivePower = zeros(size(Generation));
InverterRealPower = zeros(size(Generation));
InverterReactivePowerHacked = zeros(size(Generation));
InverterRealPowerHacked = zeros(size(Generation));

InverterRateOfChangeLimit = 100; %rate of change limit - currently unused
InverterRateOfChangeActivate = 0; %rate of change limit - currently unused

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

FilteredVoltage = zeros(size(Generation));
FilteredVoltageCalc = zeros(size(Generation));
InverterLPF = 1;
V0=SlackBusVoltage*ones(TotalTimeSteps,1);

% Initialize adaptive controller output
upk = zeros(size(IntermediateOutput_vqvp));
uqk = upk;

%% Run Simulation

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
            
            % Inverter (not compromised)
            [InverterReactivePower(ksim,knode),InverterRealPower(ksim,knode),...
            FilteredVoltage(ksim:TotalTimeSteps,knode), FilteredVoltageCalc(ksim,knode)] = ...
            inverter_model(FilteredVoltage(ksim-1,knode),...
            SolarGeneration_vqvp(ksim,knode),...
            abs(V_vqvp(knode,ksim)),abs(V_vqvp(knode,ksim-1)),...
            VBP(knode,:,ksim),TimeStep,InverterLPF,Sbar(ksim,knode),...
            InverterRealPower(ksim-1,knode),InverterReactivePower(ksim-1,knode),...
            InverterRateOfChangeLimit,InverterRateOfChangeActivate,...
            ksim,Delay_VoltageSampling(knode)); 
            
            % Inverter (hacked)
            [InverterReactivePowerHacked(ksim,knode),InverterRealPowerHacked(ksim,knode),...
            FilteredVoltage(ksim:TotalTimeSteps,knode), FilteredVoltageCalc(ksim,knode)] = ...
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
            [upk(ksim,knode)] = adaptive_control(Delay_VBPCurveShift(knode), ...
                kp, IntermediateOutput_vqvp(ksim,knode), ...
                IntermediateOutput_vqvp(ksim-Delay_VBPCurveShift(knode)+1,knode), ...
                upk(ksim-Delay_VBPCurveShift(knode)+1,knode), ThreshHold_vqvp, ...
                FilteredOutput_vqvp(ksim,knode));

            [uqk(ksim,knode)] = adaptive_control(Delay_VBPCurveShift(knode), ...
                kq, IntermediateOutput_vqvp(ksim,knode), ...
                IntermediateOutput_vqvp(ksim-Delay_VBPCurveShift(knode)+1,knode), ....
                uqk(ksim-Delay_VBPCurveShift(knode)+1,knode), ThreshHold_vqvp, ...
                FilteredOutput_vqvp(ksim,knode));
     
   % CALCULATE NEW VBP FOR UNCOMPROMISED INVERTERS
            for j = ksim:TotalTimeSteps
                VBP(knode,:,j) = [VQ_start - uqk(ksim,knode),...
                 VQ_end + uqk(ksim,knode), VP_start - upk(ksim,knode), ...
                 VP_end + upk(ksim,knode)];  
             
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
        
   % CALCULATE NEW VBP FOR HACKED INVERTERS
        if ksim > 1
            for j = ksim:TotalTimeSteps
                VBPHacked(knode,:,j) = VBPHacked(knode,:,j-1); 
            end 
        end 
   end
end

%% Figures
% t1 = datetime(2017,8,0,0,0,0);
% t_datetime = t1 + minutes(Time);
% t_hacktime = t1 + minutes(TimeOfHack);

node = 8; % node of interest for graphs

% Substation P/Q
P0_vqvp = real(S_vqvp(1,:)); 
Q0_vqvp = imag(S_vqvp(1,:));

f1 = figure(1);
subplot(2,1,1)
plot(1:501 , P0_vqvp, 'b','LineWidth',1.5)
title('Real Power (W) From Substation')
xlim([3 499])
subplot(2,1,2)
plot(1:501 , Q0_vqvp, 'r','LineWidth',1.5)
title('Reactive Power (VA) From Substation')
xlim([3 499])

% VBP for hacked and uncompromised inverters, over time
f2 = figure(2);
subplot(2,1,1)
plot(1:501 , squeeze(VBP(node,1,:)), 1:501, squeeze(VBP(node,2,:)), ...
    1:501, squeeze(VBP(node,3,:)), 1:501, squeeze(VBP(node,4,:)), ...
    'LineWidth',1.5)
title('VBP for Node 3')
legend({'V1', 'V2', 'V3', 'V4'},'FontSize',12)
xlim([3 499])
subplot(2,1,2)
plot(1:501 , squeeze(VBPHacked(node,1,:)), 1:501, squeeze(VBPHacked(node,2,:)), ...
    1:501, squeeze(VBPHacked(node,3,:)), 1:501, squeeze(VBPHacked(node,4,:)), ...
    'LineWidth',1.5)
title('VBPHacked for Node 3')
legend({'V1', 'V2', 'V3', 'V4'},'FontSize',12)
ylim([1.005 1.035])
xlim([3 499])

% Voltage seen at node 8
f3 = figure(3);
plot(1:501 , abs(V_vqvp(node,:)), 'r','LineWidth',1.5)
title(['Voltage Magnitude, node: 3'])
xlabel('time (hours)')
ylabel('volts (pu)')
xlim([3 499])

% Filtered voltage seen by inverter (note this is different filtering than
% from what's done by the observer
f4 = figure(4);
hold on
plot(1:501, FilteredVoltage(:,node), 'b','LineWidth',1.5 )
plot(1:501, FilteredVoltageCalc(:,node), 'r','LineWidth',1.5 )
hold off
title(['Inverter: LP Filter of voltage magnitude from inverter, node: 3'])
xlabel('time (hours)')
legend({'Filtered voltage used','Filtered voltage calculated'},'FontSize',12);
xlim([3 499])

% Adaptive controller input and output
f5 = figure(5);
subplot(2,1,1)
plot(1:501, upk(:,node), 1:501, uqk(:,node),'LineWidth',1.5 )
title(['Adaptive controller output, node: 3'])
legend({'upk','uqk'},'FontSize',12);
xlim([3 499]) 
subplot(2,1,2)
hold on
plot(1:501, IntermediateOutput_vqvp(:,node), 'r','LineWidth',1.5 )
plot(1:501, FilteredOutput_vqvp(:,node), 'y','LineWidth',1.5 )
plot(1:501, ThreshHold_vqvp*ones(1,TotalTimeSteps), 'm','LineWidth',1.5 )
hold off 
title(['Adaptive controller input, node: 3'])
xlabel('time (hours)')
legend({'Intermediate output', 'filtered output', 'threshold'},'FontSize',12);
xlim([3 499])

% Voltage observer inputs and outputs
f6 = figure(6);
plot(1:501, FilteredOutput_vqvp(:,node), 'LineWidth',1.5 )
legend({'filtered output'},'FontSize',12);
title(['Observer outputs'])
xlim([3 499]) 

% Inverter P/Q for hacked and uncompromised inverters (node 8)
f7 = figure(7);
subplot(2,1,1)
plot(1:501, InverterReactivePower(:,node), 1:501, InverterReactivePowerHacked(:,node), 'LineWidth',1.5) 
title(['Inverter reactive power - Node 8'])
legend({'inverter reactive power- not hacked', 'inverter reactive power - hacked'},'FontSize',12);
xlim([3 499]) 
subplot(2,1,2)
plot(1:501, InverterRealPower(:,node), 1:501, InverterRealPowerHacked(:,node), 'LineWidth',1.5)
title(['Inverter real power - Node 8'])
legend({'inverter real power- not hacked', 'inverter real power - hacked'},'FontSize',12);
xlim([3 499])

% Inverter P/Q for hacked and uncompromised inverters (node 12)
f8 = figure(8);
subplot(2,1,1)
plot(1:501, InverterReactivePower(:,12), 1:501, InverterReactivePowerHacked(:,12), 'LineWidth',1.5) 
title(['Inverter reactive power - Node 12'])
legend({'inverter reactive power- not hacked', 'inverter reactive power - hacked'},'FontSize',12);
xlim([3 499]) 
subplot(2,1,2)
plot(1:501, InverterRealPower(:,12), 1:501, InverterRealPowerHacked(:,12), 'LineWidth',1.5)
title(['Inverter real power - Node 12'])
legend({'inverter real power- not hacked', 'inverter real power - hacked'},'FontSize',12);
xlim([3 499])

% Sum of loads
f9 = figure(9);
plot(1:501, sum(Load'), 'r', 'LineWidth',1.5)
title(['Sum of Loads'])
xlabel('time (hours)')
ylabel('VA')
legend('loads')
xlim([3 499])

% Inverter P for each node
f10 = figure(10);
plot(1:501, InverterRealPower(:,3) + InverterRealPowerHacked(:,3), ...
    1:501, InverterRealPower(:,6) + InverterRealPowerHacked(:,6), ...
    1:501, InverterRealPower(:,7) + InverterRealPowerHacked(:,7), ...
    1:501, InverterRealPower(:,8) + InverterRealPowerHacked(:,8), ...
    1:501, InverterRealPower(:,10) + InverterRealPowerHacked(:,10), ...
    1:501,InverterRealPower(:,11) + InverterRealPowerHacked(:,11), ...
    1:501,InverterRealPower(:,12) + InverterRealPowerHacked(:,12),...
    1:501, InverterRealPower(:,13) + InverterRealPowerHacked(:,13), 'LineWidth',1.5)
title(['Inverter real power output for each load - combined (hacked+not hacked)'])
legend({'Node 3', 'Node 6', 'Node 7', 'Node 8', 'Node 10', 'Node 11', 'Node 12', 'Node 13'},'FontSize',12);
xlim([3 499])
