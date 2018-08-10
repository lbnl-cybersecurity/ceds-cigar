clc;
clear;
close all;

%% TUNING KNOBS - ADJUST (MAIN) PARAMETERS HERE 
% Note that there are other tunable parameters in the code, but these are
Sbase=1;
LoadScalingFactor = 2; 
GenerationScalingFactor = 3; 
SlackBusVoltage = 1.05; 
NoiseMultiplyer=01;

% Set simulation analysis period
StartTime = 40000; 
EndTime = 40900; 

% Set hack parameters
TimeStepOfHack = 50;
PercentHacked = [0.5 0 1.0 0 0 0.5 0 0.5 0 0 0 0.5 0.5 0 ...
                 0 0 1.0 0 0 0.5 0 0 0 0 0 0 0.5 0 0.5 1 0];

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
Delay_VoltageSampling = [10 10  10 10 10  10  10  10  10  10  10  10  10]; 
Delay_VBPCurveShift =   [120 120 120 120 120 120 120 120 120 120 120 120 120]; 

% Set observer voltage threshold
ThreshHold_vqvp = 0.25;
power_factor=0.9;
pf_converted=tan(acos(power_factor));
Number_of_Inverters = 13;
% The following variable allows to run the simulation without any inverter
SimulateInverterHack=01;
disp('Value Initializtion Done.')


%% Error Checking of the input data
if (EndTime<StartTime || EndTime<0 || StartTime <0)
    error('Setup Simulation Times Appropriately.')
end
if (NoiseMultiplyer<0)
    error('Setup Noise Multiplyer Correctly.')
end
%% Load the components related to OpenDSS
[DSSObj, DSSText, gridpvpath] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSMon=DSSCircuit.Monitors;
DSSText.command = 'Clear';

%% QSTS Simulation
DSSText.command = 'Compile C:\feeders\feeder34_B_NR\feeder34_B_NR.dss';
% Easy way to change parameters for all the loads , making them ZIPV
% The last parameter refers to the minimum voltage threshhold
% DSSText.Command= 'BatchEdit Load..* Model=8';
% DSSText.Command='BatchEdit Load..* ZIPV=(0.2,0.05,0.75,0.2,0.05,0.75,0.6)';

DSSSolution.Solve();
if (~DSSSolution.Converged)
    error('Initial Solution Not Converged. Check Model for Convergence');
else
    disp('Initial Model Converged. Proceeding to Next Step.')
    % Doing this solve command is required for GridPV, that is why the monitors
    % go under a reset process
    DSSMon.ResetAll;
    setSolutionParams(DSSObj,'daily',1,1,'off',1000000,30000);
    % Easy process to get all names and count of loads, a trick to avoid
    % some more lines of code
    TotalLoads=DSSCircuit.Loads.Count;
    AllLoadNames=DSSCircuit.Loads.AllNames;
    disp('OpenDSS Model Compliation Done.')
end

%% Retrieving the data from the load profile
TimeResolutionOfData=10; % resolution in minute
% Get the data from the Testpvnum folder
% Provide Your Directory
FileDirectoryBase='C:\feeders\testpvnum10\';
QSTS_Time = 0:1440; % This can be changed based on the available data
% TotalTimeSteps=length(QSTS_Time);
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

disp('Reading Data for Pecan Street is done.')
%% Interpolate to change data from minutes to seconds
disp('Starting Interpolation...')
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
    Load(:,i) = Load(:,i) + NoiseMultiplyer*Noise(:,i);
end 

if (NoiseMultiplyer>0)
    disp('Load Interpolation has been done. Noise was added to the load profile.') 
else
    disp('Load Interpolation has been done. No Noise was added to the load profile.') 
end
MaxGenerationPossible = max(Generation); % Getting the Maximum possible Generation for each Load 
%% Initializing the inverter models 

if Number_of_Inverters> TotalLoads
    exit('Not Supported Right now');
end
% Creating an array of Inverter objects
InverterArray(1:Number_of_Inverters)=Inverter;
for i = 1:Number_of_Inverters
    InverterArray(i).Name=AllLoadNames{i+5};
    InverterArray(i).Delay_VoltageSampling=Delay_VoltageSampling(i);
    InverterArray(i).Delay_VBPCurveShift=Delay_VBPCurveShift(i);
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
% Getting Bus Names for Inverters 
InverterBusNames= {InverterArray.Name};
Generation=Generation(:,1:Number_of_Inverters);
disp('Array of Inverter objects has been created.')


%%  Power Flow with VQVP Control using OpenDSS 
% Initialize feeder states with control case
V_vqvp = zeros(Number_of_Inverters,TotalTimeSteps);
S_vqvp = zeros(Number_of_Inverters,TotalTimeSteps);
IterationCounter_vqvp = zeros(Number_of_Inverters,TotalTimeSteps);
PowerEachTimeStep_vqvp = zeros(Number_of_Inverters,3);

% Percentage of irradiation "seen by" uncompromised and hacked inverters
SolarGeneration_vqvp_TOTAL = Generation * GenerationScalingFactor;
SolarGeneration_vqvp = SolarGeneration_vqvp_TOTAL;
SolarGeneration_vqvp_hacked = SolarGeneration_vqvp_TOTAL;

% Lets not use the advanced 
for t = TimeStepOfHack:TotalTimeSteps
    for i= 1:Number_of_Inverters
        SolarGeneration_vqvp(t,i) = SolarGeneration_vqvp_TOTAL(t,i) .* (1- PercentHacked(i));
        SolarGeneration_vqvp_hacked(t,i) = SolarGeneration_vqvp_TOTAL(t,i) .* PercentHacked(i);
    end 
end 

%% Setting up the maximum power capability
% Sbar_max =  MaxGenerationPossible * GenerationScalingFactor;
% Sbar_max=Sbar_max(1:Number_of_Inverters);
% 
% % duplicating array using the matrix usage of MATLAB 
% Sbar=ones(size(Generation)).*Sbar_max;
% 
% Sbar(TimeStepOfHack:TotalTimeSteps,:) =ones(length(TimeStepOfHack:TotalTimeSteps),Number_of_Inverters).*Sbar_max.*(1-[InverterArray.PercentHacked]);
% 

Sbar_max =  MaxGenerationPossible * GenerationScalingFactor;
Sbar = zeros(size(Generation));
SbarHacked = Sbar_max .* PercentHacked;

for t = 1:TotalTimeSteps
    Sbar(t,:) = Sbar_max(1:Number_of_Inverters);
end

for t = TimeStepOfHack:TotalTimeSteps
    Sbar(t,:) = Sbar_max(1:Number_of_Inverters).*(1-[InverterArray.PercentHacked]);
end 

%% Voltage Observer Parameters and related variable initialization

InverterReactivePower = zeros(size(Generation));
InverterRealPower = zeros(size(Generation));
InverterReactivePowerHacked = zeros(size(Generation));
InverterRealPowerHacked = zeros(size(Generation));
FilteredOutput_vqvp = zeros(TotalTimeSteps,TotalLoads);
IntermediateOutput_vqvp=zeros(TotalTimeSteps,TotalLoads);
Epsilon_vqvp = zeros(TotalTimeSteps,TotalLoads);
upk = zeros(TotalTimeSteps,TotalLoads);
uqk = zeros(TotalTimeSteps,TotalLoads);
FilteredVoltage = zeros(size(Generation));
FilteredVoltageCalc = zeros(size(Generation));
%%
% Initialize VBP for hacked and uncompromised inverters
VBP = [nan*ones(Number_of_Inverters,1,TotalTimeSteps), nan*ones(Number_of_Inverters,1,TotalTimeSteps), ...
       nan*ones(Number_of_Inverters,1,TotalTimeSteps), nan*ones(Number_of_Inverters,1,TotalTimeSteps)];
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
disp('Setting up of the solution variables are done.')
%% OpenDSS Parameters
setSourceInfo(DSSObj,{'source'},'pu',SlackBusVoltage);
for ksim =1:TotalTimeSteps
    
    if (ksim>1)
        IntermediateLoadValue=Load(ksim,1:Number_of_Inverters)+SimulateInverterHack*InverterRealPower(ksim-1,:)...
                            + SimulateInverterHack*InverterRealPowerHacked(ksim-1,:);
%         IntermediateLoadValue=IntermediateLoadValue; % To convert to KW
        setLoadInfo(DSSObj,InverterBusNames,'kw',IntermediateLoadValue); % To convert to KW
        setLoadInfo(DSSObj,InverterBusNames,'kvar',pf_converted*Load(ksim,1:Number_of_Inverters)+SimulateInverterHack*InverterReactivePower(ksim-1,:)...
                                +SimulateInverterHack*InverterReactivePowerHacked(ksim-1,:));
        setLoadInfo(DSSObj,{AllLoadNames{Number_of_Inverters+1:end}},'kw',(Load(ksim,Number_of_Inverters+1:end))); % To convert to KW
        setLoadInfo(DSSObj,{AllLoadNames{Number_of_Inverters+1:end}},'kvar',pf_converted*(Load(ksim,Number_of_Inverters+1:end)));
    else
        setLoadInfo(DSSObj,AllLoadNames,'kw',(Load(ksim,:))); % To convert to KW
        setLoadInfo(DSSObj,AllLoadNames,'kvar',pf_converted*(Load(ksim,:)));
  
    end
   % Solving the Power Flow
    DSSSolution.Solve();
    if (~DSSSolution.Converged)
        error(strcat('Solution Not Converged at Step_', string(ksim)))
    end
    % Retrieving the Voltage Information
    InverterInfo=getLoadInfo(DSSObj,InverterBusNames);
    V_vqvp(:,ksim)=[InverterInfo.voltagePU];
    
    if(ksim > 1 && ksim < TotalTimeSteps)
      for knode = 1:Number_of_Inverters
        [InverterReactivePower(ksim,knode),InverterRealPower(ksim,knode),...
            FilteredVoltage(ksim:TotalTimeSteps,knode), FilteredVoltageCalc(ksim,knode)]=...
            voltvarvoltwatt(InverterArray(knode),FilteredVoltage(ksim-1,knode),...
            SolarGeneration_vqvp(ksim,knode),...
            abs(V_vqvp(knode,ksim)),abs(V_vqvp(knode,ksim-1)),...
            VBP(knode,:,ksim),Sbar(ksim,knode),...
            InverterRealPower(ksim-1,knode),InverterReactivePower(ksim-1,knode),ksim);
        
        [InverterReactivePowerHacked(ksim,knode),InverterRealPowerHacked(ksim,knode),...
            FilteredVoltage(ksim:TotalTimeSteps,knode), FilteredVoltageCalc(ksim,knode)]=...
            voltvarvoltwatt(InverterArray(knode),FilteredVoltage(ksim-1,knode),...
            SolarGeneration_vqvp(ksim,knode),...
            abs(V_vqvp(knode,ksim)),abs(V_vqvp(knode,ksim-1)),...
            VBP(knode,:,ksim),Sbar(ksim,knode),...
            InverterRealPowerHacked(ksim-1,knode),InverterReactivePowerHacked(ksim-1,knode),ksim);
        
        
        [FilteredOutput_vqvp(ksim,knode),IntermediateOutput_vqvp(ksim,knode), ... 
                Epsilon_vqvp(ksim,knode)] = voltageobserver(InverterArray(knode),V_vqvp(knode,ksim), ...
                V_vqvp(knode,ksim-1), IntermediateOutput_vqvp(ksim-1,knode), ...
                Epsilon_vqvp(ksim-1,knode), FilteredOutput_vqvp(ksim-1,knode));
    
         if mod(ksim, InverterArray(knode).Delay_VBPCurveShift) == 0 
            [upk(ksim,knode)] = adaptivecontrolreal(InverterArray(knode), ...
                IntermediateOutput_vqvp(ksim,knode), ...
                IntermediateOutput_vqvp(ksim-Delay_VBPCurveShift(knode)+1,knode), ...
                upk(ksim-Delay_VBPCurveShift(knode)+1,knode), ...
                FilteredOutput_vqvp(ksim,knode));
            
            [uqk(ksim,knode)] = adaptivecontrolreactive(InverterArray(knode), ...
                IntermediateOutput_vqvp(ksim,knode), ...
                IntermediateOutput_vqvp(ksim-Delay_VBPCurveShift(knode)+1,knode), ...
                uqk(ksim-Delay_VBPCurveShift(knode)+1,knode), ...
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
legend('Real Power (kW)','Reactive Power (kVAr)')
xlim([1 length(time)]);
title('Power From the Substation')


%%
% Plot the movement of VBP

for node=1:Number_of_Inverters
    plot(time , squeeze(VBP(node,1,:)), time, squeeze(VBP(node,2,:)), ...
    time, squeeze(VBP(node,3,:)), time, squeeze(VBP(node,4,:)), ...
    'LineWidth',1.5)
    hold on
    
end
xlim([1  501])

figure
plot(time,V_vqvp(:,:),'linewidth',1.05)
ylabel('Per Unit Voltage')
xlim([1 length(time)]);
