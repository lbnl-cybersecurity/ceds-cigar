function [ Voltage,Power ] = FBS(NodeVoltageToPlot,SlackBusVoltage,IncludeSolar)
% This function is a version of running the FBS, suitable for running from a python code

% Convert the variables to double as MATLAB does not support integer
% airthmatic
SlackBusVoltage=double(SlackBusVoltage);
IncludeSolar = double(IncludeSolar);
LoadScalingFactor=0.9*1.5;
GenerationScalingFactor=2;
% SlackBusVoltage=1.0;
power_factor=0.9;
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

FileDirectoryBase='C:\Users\Sy-Toan\ceds-cigar\LBNL_Simulations\testpvnum10\';
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
 
% NodeVoltageToPlot=634;
node = NodeList==NodeVoltageToPlot;
Voltage=abs(V_nc(node,:));
Power=P0_nc/1000;



