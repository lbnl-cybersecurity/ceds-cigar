% Author : Shammya Saha (sssaha@lbl.gov)
% Date : 05/22/2018
clc;
clear;
close all;

%% Constants 
% These constants are used to create appropriate simulation scenario
LoadScalingFactor=0.9*1.5;
GenerationScalingFactor=2;

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

Z = Z_in_ohm/Zbase; % Here Z represents Z in per unit
% Why we are changing the Z(2) data
% Z(2) = Z(2)+0.01+0.08j; % Adding Transformer Reactance

%% Load Data Pertaining to Loads to create a profile
TimeResolutionOfData=10; % resolution in minute
% Get the data from the Testpvnum folder
% FileDirectoryBase=strcat('testpvnum',num2str(TimeResolutionOfData),'/');
FileDirectoryBase='C:\Users\shamm\Dropbox (ASU)\ASU\Microgrid Controller\CIGAR\CEDS_CIGAR\LBNL Simulations\testpvnum10\';
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
ConstantPowerFraction = 0.75;  %constant power fraction of the load
ConstantCurrentFraction = 0.05; %constant current fraction of the load
ConstantImpedanceFraction = 0.2; %constant impedance fraction of the load
ZIP_demand = zeros(TotalTimeSteps,IeeeFeeder,3); 
power_factor=0.9;
for node = 2:IeeeFeeder
    ZIP_demand(:,node,:) = [ConstantPowerFraction*Load(:,node), ConstantCurrentFraction*Load(:,node), ...
    ConstantImpedanceFraction*Load(:,node)]*(1 + 1i*tan(acos(power_factor))); % Q = P tan(theta)
end


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

% yk = yk_nc;
% psik_nc = yk;
% psik = yk;
% epsilonk_nc = yk;
% epsilonk = yk;
%%  Power Flow For No Control Case
V0=1.00*ones(TotalTimeSteps,1);

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
   PowerEachTimeStep_nc(knode,:) = [ZIP_demand(ksim,knode,1) - 0*SolarGeneration_NC(ksim,knode),...
            ZIP_demand(ksim,knode,2),ZIP_demand(ksim,knode,3)];    
   end
   % Doing the FBS power flow
   [V_nc(:,ksim),Irun,S_nc(:,ksim),IterationCounter_nc(:,ksim)] = FBSfun(V0(ksim),PowerEachTimeStep_nc,Z,FeederMap);
   
   % Observer Modeling
%    for knode=1:length(NodeList)
%         if(ksim > 1)
%             [FilteredOutput_nc(ksim,knode),IntermediateOutput_nc(ksim,knode),Epsilon_nc(ksim,knode)] = voltage_observer(V_nc(knode,ksim), ...
%                 V_nc(knode,ksim-1), IntermediateOutput_nc(ksim-1,knode), ...
%                 Epsilon_nc(ksim-1,knode), FilteredOutput_nc(ksim-1,knode),...
%                 HighPassFilterFrequency, LowPassFilterFrequency, Gain_Energy, TimeStep);
%         end
%     end
end



%% OpenDSS integration

[DSSObj, DSSText, gridpvpath] = DSSStartup;
% Load the components related to OpenDSS
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSMon=DSSCircuit.Monitors;

%Compile the Model
DSSText.command = 'Compile C:\feeders\feeder13_B_R\feeder13BR.dss';

LoadBus=NodeList(LoadList);
LoadBusNames=cell(1,8);
for i = 1:length(LoadBus)
    LoadBusNames{i} = strcat('load_',num2str(LoadBus(i)));
end
power_factor=0.9;
NodeVoltageToPlot=634;
VoltageOpenDSS=zeros(1,TotalTimeSteps);
setSourceInfo(DSSObj,[{'source'}],'pu',[1.00]);
for ksim=1:TotalTimeSteps
   setLoadInfo(DSSObj,LoadBusNames,'kw',Load(ksim,LoadList)/1000);
    setLoadInfo(DSSObj,LoadBusNames,'kvar',tan(acos(power_factor))*Load(ksim,LoadList)/1000);
    DSSSolution.Solve();
%     LineInfo = getLineInfo(DSSObj, [{'L_U_650'}]);
%     IncomingRealPower=[LineInfo.bus1PowerReal];
%     IncomingReactivePower=[LineInfo.bus1PowerReactive];
% disp(IncomingRealPower)
% disp(IncomingReactivePower)
    BusInfo=getBusInfo(DSSObj,{strcat('bus_',num2str(NodeVoltageToPlot))});
    VoltageOpenDSS(1,ksim)=BusInfo.voltagePU;
end

% 
%% %%  Power Flow with VQVP Control Case
% %Initialize feeder states with control case
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
V0=1.02*ones(TotalTimeSteps,1);
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
%    for node_iter=1:NumberOfLoads
%         knode = LoadList(node_iter);
%         if(ksim > 1)
%             [FilteredOutput_vqvp(ksim,knode),IntermediateOutput_vqvp(ksim,knode),Epsilon_vqvp(ksim,knode)] = voltage_observer(V_vqvp(knode,ksim), ...
%                 V_vqvp(knode,ksim-1), IntermediateOutput_vqvp(ksim-1,knode), ...
%                 Epsilon_vqvp(ksim-1,knode), FilteredOutput_vqvp(ksim-1,knode), HighPassFilterFrequency,...
%                 LowPassFilterFrequency, Gain_Energy, TimeStep);
%         end
%          if(FilteredOutput_vqvp(ksim,knode) >= ThreshHold_vqvp)
%             %we are, so re-dispatch smart inverter settings
%             %comment code below to see unstable results
% %              VBP(knode,:) = [1.025,1.04,1.04,1.05];
%         end
%         
%    end
end
% 
% 
% 
% %%  Calculate feeder head powers
% P0_nc = real(S_nc(1,:));
% Q0_nc = imag(S_nc(1,:));
% 
% P0_vqvp = real(S_vqvp(1,:));
% Q0_vqvp = imag(S_vqvp(1,:));
% 
% 
% 
% %% Figures - overlay
close all
t1 = datetime(2017,8,0,0,0,0);
t_datetime = t1 + minutes(Time);
% 
% 
node = find(NodeList==NodeVoltageToPlot);
f1 = figure(1);
set(f1,'Units','Inches');
pos = get(f1,'Position');
set(f1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime , abs(V_nc(node,:)), 'b','LineWidth',1.5)
hold on;
plot(t_datetime , VoltageOpenDSS, 'r','LineWidth',1.5)
legend('FBS','OpenDSS')
% hold off
% % datetick('x','HH:MM')
% datetick('x','HH:MM')
% title(['Voltage Magnitude, node: ', num2str(NodeList(node))],'FontSize',14)
% legend({'feedthrough','VV-VW control'},'FontSize',14);
% xlabel('time (hours)')
% ylabel('volts (pu)')
% % set(gca,'FontSize',14,'FontName','Times New Roman')
% % set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
% %     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
% %     datestr(datenum(2017,8,0,0:4:24,0,0),15))
% 
% 
% f2 = figure(2);
% set(f2,'Units','Inches');
% pos = get(f2,'Position');
% set(f2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
% plot(t_datetime,FilteredOutput_nc(:,node), 'b','LineWidth',1.5)
% hold on
% plot(t_datetime, FilteredOutput_vqvp(:,node), 'r','LineWidth',1.5 )
% hold off
% datetick('x','HH:MM')
% title(['Observer Output, node: ', num2str(NodeList(node))])
% legend({'feedthrough','VV-VW control'},'FontSize',14);
% xlabel('time (hours)')
% % set(gca,'FontSize',14,'FontName','Times New Roman')
% % set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
% %     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
% %     datestr(datenum(2017,8,0,0:4:24,0,0),15))
% ylabel('instability (V^2) (pu)')
% 
% 
% f3 = figure(3);
% set(f3,'Units','Inches');
% pos = get(f3,'Position');
% set(f3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
% plot(t_datetime,P0_nc, 'b','LineWidth',1.5)
% hold on
% plot(t_datetime,P0_vqvp, 'r','LineWidth',1.5)
% hold off
% datetick('x','HH:MM')
% legend({'feedthrough','VV-VW control'},'FontSize',14,'Location','SouthEast');
% % set(gca,'FontSize',14,'FontName','Times New Roman')
% % set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
% %     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
% %     datestr(datenum(2017,8,0,0:4:24,0,0),15))
% title('Substation Real Power')
% xlabel('time (hours)')
% ylabel('Watts (pu)')
% 
% f4 = figure(4);
% set(f4,'Units','Inches');
% pos = get(f4,'Position');
% set(f4,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
% plot(t_datetime,Q0_nc, 'b','LineWidth',1.5)
% hold on
% plot(t_datetime,Q0_vqvp, 'r','LineWidth',1.5)
% hold off
% datetick('x','HH:MM')
% title('Substation Reactive Power')
% xlabel('time (hours)')
% ylabel('VARs (pu)')
% legend({'feedthrough','VV-VW control'},'FontSize',14);
% % set(gca,'FontSize',14,'FontName','Times New Roman')
% % set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
% %     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
% %     datestr(datenum(2017,8,0,0:4:24,0,0),15))
% 
% 
% f5 = figure(5);
% set(f5,'Units','Inches');
% pos = get(f5,'Position');
% set(f5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
% plot(t_datetime,InverterRealPower(:,4), 'b','LineWidth',1.5)
% hold on
% plot(t_datetime,InverterReactivePower(:,4), 'r','LineWidth',1.5)
% hold off
% datetick('x','HH:MM')
% title(['DER Power Output, node: ', num2str(NodeList(4))])
% xlabel('time (hours)')
% ylabel('pu')
% legend({'active power','reactive power'},'FontSize',14);
% % set(gca,'FontSize',14,'FontName','Times New Roman')
% % set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
% %     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
% %     datestr(datenum(2017,8,0,0:4:24,0,0),15))