clc;
clear;
close all;

%% TUNING KNOBS - ADJUST (MAIN) PARAMETERS HERE 
% Note that there are other tunable parameters in the code, but these are
% the main ones
Vbase = 4.16e3; %4.16 kV
Sbase = 1; % Set to something like '500 kVA' to use per unit
Zbase = Vbase^2/Sbase;
Ibase = Sbase/Vbase;
% Set load factors and slack voltage
LoadScalingFactor = 2000; 
GenerationScalingFactor = 70; 
SlackBusVoltage = 1.02; 

% Set simulation analysis period
StartTime = 40000; 
EndTime = 40500; 

% Set hack parameters
TimeStepOfHack = 50;
PercentHacked = [0 0 1.0 0 0 0.5 0.5 0.5 0 0.5 0.5 0.5 0.5 ...
    0 0 1.0 0 0 0.5 0.5 0.5 0 0.5 0.5 0.5 0.5 0 0 0 0 0];

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

%%

setSolutionParams(DSSObj,'daily',1,1,'off',1000000,30000);
% Easy process to get all names and count of loads 
TotalLoads=DSSCircuit.Loads.Count;
AllLoadNames=DSSCircuit.Loads.AllNames;
% Creating a random distribution between 0 and 1, This part will get
% changed whien you will try to incorporate the actual load profile
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
power_factor=0.9;
pf_converted=tan(acos(power_factor));


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

upk = zeros(size(IntermediateOutput_vqvp));
uqk = upk;

%% Voltage Observer Parameters and related variable initialization
LowPassFilterFrequency = 0.1; % Low pass filter
HighPassFilterFrequency = 1; % high pass filter
Gain_Energy = 1e5;
TimeStep=1;
FilteredOutput_vqvp = zeros(TotalTimeSteps,NumberOfNodes);
IntermediateOutput_vqvp=zeros(TotalTimeSteps,NumberOfNodes);
Epsilon_vqvp = zeros(TotalTimeSteps,NumberOfNodes);
%%

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

% BaseVoltage=24.9*1000/sqrt(3);
% DSSMon.Name='solar 01 VI';
% Voltage_01= ExtractMonitorData(DSSMon,1,BaseVoltage);
% figure
% plot(time,Voltage_01 / 3,'k', 'linewidth',1.5)
% xlim([1 length(time)]);
% title ('Bus Voltage (Average)')






