% Author : Shammya Saha
% sssaha@lbl.gov / sssaha@asu.edu

clc;
clear;
close all;

%% Load the components related to OpenDSS
[DSSObj, DSSText, gridpvpath] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
DSSText.command = 'Clear';

%% QSTS Simulation
% Total number of simulation steps
NumberofSteps=100;
voltage_fromTransformer=zeros(1,NumberofSteps);
voltage_frombus=zeros(1,NumberofSteps);
% Compiling the model and setting the solution parameters
DSSText.command = 'Compile C:\feeders\34BusLTCInverse\Radial34BusLTCInverse.dss';
setSolutionParams(DSSObj,'daily',1,1,'OFF',1000000,30000);
for i = 1: NumberofSteps
%     Randomly selecting the tap value between 1 to 16
    r = randi(16,1,1,'uint32');
%     Setting the associated tap value for the transformer, The OpenDSS
%     control is off, but the regulator is still enabled
%     To change the transformer tap you need to change the tap of the
%     associated regulator, 
    setRegInfo(DSSObj,{'LTC-t_02'},'tapnumber',r);
%     Solving the Power Flow
    DSSSolution.Solve();
%     Get the secondary voltage of the transformer 
    Transformer=getTransformerInfo(DSSObj,{'t 02'});
    voltage_fromTransformer(i)=Transformer.bus2VoltagePU;
%     The same information can be retrieved using the businfo function
    BusInfo=getBusInfo(DSSObj,{'bus_850'});
    voltage_frombus(i)=BusInfo.voltagePU;
end


%% Plotting Monitors Data
DSSMon=DSSCircuit.Monitors;
DSSMon.Name='tapMonitor';
fprintf('\nMonitor Name: %s\n',DSSMon.Name);
Headers=DSSMon.Header;
fprintf('Header Name: %s\n',Headers{:} );
time=ExtractMonitorData(DSSMon,0,DSSSolution.StepSize);
tap= ExtractMonitorData(DSSMon,1,1);
figure
% The Voltage fluctuation happens because of the change of load profile
% The load profile can be ignored by changing some lines in the OpenDSS
% model directly
plot(time,tap,'r',time,voltage_fromTransformer,'k',time,voltage_frombus,'b','linewidth',1.5);
legend('Tap Position (pu)')
xlim([1 length(time)]);


