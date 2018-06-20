%% getRegInfo
% Gets the information for all regulators in the circuit
%
%% Syntax
%  Regulators = getRegInfo(DSSCircObj)
%  Regulators = getRegInfo(DSSCircObj, regNames)
%
%% Description
% Function to get the information about the regulators in the circuit and
% return a structure with the information. If the optional input of
% regNames is filled, the function returns information for the specified
% subset of regulators, excluding the miscellaneous parameters mentioned in the
% outputs below.
%
%% Inputs
% * *|DSSCircObj|* - link to OpenDSS active circuit and command text (from DSSStartup)
% * *|regNames|* - optional cell array of regulator names to get information for
%
%% Outputs
% *|Regulators|* is a structure with all the parameters for the
% regulatorss in the active circuit.  Fields are:
%
% * _name_ - Name of the regulator.
% * _busName_ - Name of the associated bus.
% * _numPhases_ - Number of phases associated with the load.
% * _enabled_ - {1|0} indicates whether this element is enabled in the simulation.
% * _coordinates_ - Coordinates for the load's bus, obtained from getBusInfo.
% * _distance_ - Line distance from the load's bus to the substation, obtained from getBusInfo.
% * _current_ - average phase current
% * _phaseVoltages_ - Value of voltage magnitudes calculated from
% the complex voltage returned by OpenDSS. Length is always 3,
% returning 0 for phases not on the bus
% * _phaseVoltagesPU_ - Per-unit value of voltage magnitudes calculated from
% the complex per-unit voltage returned by OpenDSS. Length is always 3,
% returning 0 for phases not on the bus.
% * _voltage_, _voltagePU_, _voltagePhasorPU_, _phaseVoltages_, _phaseVoltagePhasors_, ... 
% _phaseVoltagePhasorsPU_, _phaseVoltagesLL_, _phaseVoltagesLLPU_, _voltageLL_, _voltageLLPU_ - voltages and voltage phasors
% * _seqVoltages_, _cplxVoltages_, _seqCurrents_, _cplxSeqCurrents_ - zero, positive, and negative sequence voltages and currents magnitude or complex phasors
% * _phasePowerReal_ - 3-element array of the real components of each
% phase's complex power injected by generator. Phases that are not present will return 0.
% * _phasePowerReactive_ - 3-element array of the imaginary components of each
% phase's complex power injected by generator. Phases that are not present will return 0.
% * _powerReal_ - Total _phasePowerReal_.
% * _powerReactive_ - Total _phasePowerReactive_.
% * _losses_ - total real and imaginary power losses
% * _phaseLosses_ - real and imaginary power losses
% * _MaxTapChange_ - Maximum Tap Change allowed in one control iteration.
% * _delay_ - delay of the regulator.
% * _tapdelay_ - delay of the arm of the regulator.
% * _TapNumber_ - Current Tap position of the regulator.
% * _Transformer_ - The assciated transformer name of the regulator

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get Regulators 
function Regulators = getRegInfo(DSSCircObj, varargin)
%% Parse inputs
p = inputParser; %setup parse structure
p.addRequired('DSSCircObj', @isinterfaceOpenDSS);
p.addOptional('regNames', 'noInput', @iscellstr);

p.parse(DSSCircObj, varargin{:}); %parse inputs

allFields = fieldnames(p.Results); %set all parsed inputs to workspace variables
for ii=1:length(allFields)
    eval([allFields{ii}, ' = ', 'p.Results.',allFields{ii},';']);
end

try
%% Define the circuit
DSSCircuit = DSSCircObj.ActiveCircuit;

if strcmp(regNames, 'noInput')
    regNames = DSSCircuit.RegControls.AllNames;
end

Regulators = struct('name',regNames);
disp(regNames)
% Return if there are no loads in the circuit
if strcmp(regNames,'NONE')
    return;
end

% Get voltages bases
kVBases = DSSCircuit.Settings.VoltageBases;
kVBases = [kVBases kVBases/sqrt(3)]; % kvBases are only LL, adding LN


%% Get Regulator Info    
for ii=1:length(Regulators)
    DSSCircuit.SetActiveElement(strcat('RegControl.', Regulators(ii).name));
    if ~strcmpi(['RegControl.' regNames{ii}], DSSCircuit.ActiveElement.Name)
        error('RegName:notfound',sprintf('Regulator ''%s'' is not found in the circuit.  Check that this is a load in the compiled circuit.', regNames{ii}))
    end
    
    buses = DSSCircuit.ActiveElement.BusNames;
    Regulators(ii).busName = buses{1};
    
    Regulators(ii).numPhases = DSSCircuit.ActiveCktElement.NumPhases;
    
    Regulators(ii).enabled = DSSCircuit.ActiveElement.Enabled;
    if ~Regulators(ii).enabled % regulator is not enabled, so much of the active element properties will return errors
        continue;
    end
    
    numConductors = DSSCircuit.ActiveCktElement.NumConductors;
    
    nodes = DSSCircuit.ActiveElement.nodeOrder;
    nodes1 = nodes(1:numConductors);
    nodes1 = nodes1(nodes1~=0);
    
    Regulators(ii).nodes = nodes1;
    
    % Currents
    currents = DSSCircuit.ActiveCktElement.Currents; %complex currents
    currents = reshape(currents,2,[]); %two rows for real and reactive
    currents = hypot(currents(1,:),currents(2,:)); %current magnitude
    
    Regulators(ii).current = mean(currents(1:DSSCircuit.ActiveCktElement.NumPhases));
    
    % Voltages    
    DSSCircuit.SetActiveBus(Regulators(ii).busName);
    if isempty(DSSCircuit.ActiveBus.Name)
        generalName = regexprep(Regulators(ii).busName,'(\.[0-9]+)',''); %take out the phase numbers on buses if they have them
        DSSCircuit.SetActiveBus(generalName);
    end
    if isempty(DSSCircuit.ActiveBus.Name)
        error('busName:notfound',sprintf('Bus ''%s'' of Load ''%s'' is not found in the circuit.  Check that this is a bus in the compiled circuit.',Transformers(ii).bus1, Transformers(ii).name))
    end
    
    Regulators(ii).coordinates = [DSSCircuit.ActiveBus.y, DSSCircuit.ActiveBus.x];
    Regulators(ii).distance = DSSCircuit.ActiveBus.Distance;
    
    voltages = DSSCircuit.ActiveBus.Voltages; %complex voltages
    compVoltages = voltages(1:2:end) + 1j*voltages(2:2:end);
    voltages = reshape(voltages,2,[]); %two rows for real and reactive
    voltages = hypot(voltages(1,:),voltages(2,:)); %voltage magnitude
    Regulators(ii).voltage = mean(voltages(1:DSSCircuit.ActiveBus.NumNodes));
    
    voltagesPU = DSSCircuit.ActiveBus.puVoltages; %complex voltages
    compVoltagesPU = voltagesPU(1:2:end) + 1j*voltagesPU(2:2:end);
    voltagesPU = reshape(voltagesPU,2,[]); %two rows for real and reactive
    voltagesPU = hypot(voltagesPU(1,:),voltagesPU(2,:)); %voltage magnitude
    Regulators(ii).voltagePU = mean(voltagesPU(1:DSSCircuit.ActiveBus.NumNodes));
    Regulators(ii).voltagePhasorPU = mean(compVoltagesPU(1:DSSCircuit.ActiveBus.NumNodes));
    
    busPhaseVoltages = zeros(1,3);
    
    phaseVoltages = zeros(1,3);
    busPhaseVoltagesPU = zeros(1,3);
    phaseVoltagesPU = zeros(1,3);
    busPhaseVoltagePhasors = zeros(1,3);
    phaseVoltagePhasors = zeros(1,3);
    busPhaseVoltagePhasorsPU = zeros(1,3);
    phaseVoltagePhasorsPU = zeros(1,3);
    
    busPhaseVoltages(DSSCircuit.ActiveBus.Nodes) = voltages;
    phaseVoltages(nodes1) = busPhaseVoltages(nodes1);
    busPhaseVoltagesPU(DSSCircuit.ActiveBus.Nodes) = voltagesPU;
    phaseVoltagesPU(nodes1) = busPhaseVoltagesPU(nodes1);
    busPhaseVoltagePhasors(DSSCircuit.ActiveBus.Nodes) = compVoltages;
    phaseVoltagePhasors(nodes1) = busPhaseVoltagePhasors(nodes1);
    busPhaseVoltagePhasorsPU(DSSCircuit.ActiveBus.Nodes) = compVoltagesPU;
    phaseVoltagePhasorsPU(nodes1) = busPhaseVoltagePhasorsPU(nodes1);
    
    Regulators(ii).phaseVoltages = phaseVoltages;
    Regulators(ii).phaseVoltagesPU = phaseVoltagesPU;
    Regulators(ii).phaseVoltagePhasors = phaseVoltagePhasors;
    Regulators(ii).phaseVoltagePhasorsPU = phaseVoltagePhasorsPU;
    
    phaseVoltagesLN = abs(phaseVoltagePhasors);
    sngPhBus = sum(phaseVoltagesLN~=0, 2) == 1;
    
    phaseVoltagesLL = phaseVoltagesLN;
    if ~sngPhBus
        phaseVoltagesLL = abs([phaseVoltagePhasors(1) - phaseVoltagePhasors(2), ...
            phaseVoltagePhasors(2) - phaseVoltagePhasors(3), phaseVoltagePhasors(3) - phaseVoltagePhasors(1)] .* ...
            [phaseVoltagesLN(1) & phaseVoltagesLN(2), phaseVoltagesLN(2) & phaseVoltagesLN(3)...
            phaseVoltagesLN(3) & phaseVoltagesLN(1)]);
    end
    
    Regulators(ii).phaseVoltagesLL = phaseVoltagesLL;
    
    % get pu
    phaseVoltagesLLAvg = sum(phaseVoltagesLL)./sum(phaseVoltagesLL~=0);
    baseDiff = kVBases - phaseVoltagesLLAvg/1000;
    [~, ind] = min(abs(baseDiff), [], 2);
    phaseVoltagesLLPU = phaseVoltagesLL./kVBases(ind)' / 1000;
    Regulators(ii).phaseVoltagesLLPU = phaseVoltagesLLPU;
    % avg line to line voltages
    Regulators(ii).voltageLL = phaseVoltagesLLAvg;
    Regulators(ii).voltageLLPU = phaseVoltagesLLAvg/kVBases(ind)' / 1000;
    
    Regulators(ii).seqVoltages = DSSCircuit.ActiveCktElement.SeqVoltages;
    Regulators(ii).cplxSeqVoltages = DSSCircuit.ActiveCktElement.CplxSeqVoltages;
    Regulators(ii).seqCurrents = DSSCircuit.ActiveCktElement.SeqCurrents;
    Regulators(ii).cplxSeqCurrents = DSSCircuit.ActiveCktElement.CplxSeqCurrents;
    
    power = DSSCircuit.ActiveCktElement.Powers; %complex
    power = reshape(power,2,[]); %two rows for real and reactive
    
    Regulators(ii).phasePowerReal = power(1,1:DSSCircuit.ActiveCktElement.NumPhases);
    Regulators(ii).phasePowerReactive = power(2,1:DSSCircuit.ActiveCktElement.NumPhases);
    
    Regulators(ii).powerReal = sum(power(1,1:DSSCircuit.ActiveCktElement.NumPhases));
    Regulators(ii).powerReactive = sum(power(2,1:DSSCircuit.ActiveCktElement.NumPhases));
    
    losses = DSSCircuit.ActiveCktElement.Losses;
    Regulators(ii).losses = losses(1)/1000 + 1i*losses(2)/1000;
    
    losses = DSSCircuit.ActiveCktElement.PhaseLosses;
    losses = reshape(losses,2,[]);
    Regulators(ii).phaseLosses = losses(1,:) + 1i*losses(2,:);
    
end

%% Remove loads that are not enabled if no names were input to the function
condition = [Regulators.enabled]==0;
if ~isempty(varargin) && any(condition) %if the user specified the load names, return warning for that load not being enabled
    warning(sprintf('Regulator %s is not enabled\n',Regulators(condition).name));
else
    Regulators = Regulators(~condition);
end

%% Get load parameters

for ii=1:length(Regulators)
    DSSCircuit.RegControls.name = Regulators(ii).name;
    Regulators(ii).MaxTapChange = get(DSSCircuit.RegControls,'MaxTapChange');
    Regulators(ii).Delay = get(DSSCircuit.RegControls,'Delay');
    Regulators(ii).TapDelay = get(DSSCircuit.RegControls,'TapDelay');
    Regulators(ii).TapNumber = get(DSSCircuit.RegControls,'TapNumber');
    Regulators(ii).Transformer = get(DSSCircuit.RegControls,'Transformer');
   
    
    %Get additional miscellaneous parameters that are present but are currently unused in the toolbox
%     Regulators(ii).Idx = get(DSSCircuit.Loads,'Idx');
%     Regulators(ii).pctMean = get(DSSCircuit.Loads,'PctMean');
%     Regulators(ii).pctStdDev = get(DSSCircuit.Loads,'PctStdDev');
%     Regulators(ii).allocationFactor = get(DSSCircuit.Loads,'AllocationFactor');
%     Regulators(ii).Cfactor = get(DSSCircuit.Loads,'Cfactor');
%     Regulators(ii).class = get(DSSCircuit.Loads,'Class');
%     Regulators(ii).isDelta = get(DSSCircuit.Loads,'IsDelta');
%     Regulators(ii).CVRcurve = get(DSSCircuit.Loads,'CVRcurve');
%     Regulators(ii).CVRwatts = get(DSSCircuit.Loads,'CVRwatts');
%     Regulators(ii).CVRvars = get(DSSCircuit.Loads,'CVRvars');
%     Regulators(ii).daily = get(DSSCircuit.Loads,'daily');
%     Regulators(ii).duty = get(DSSCircuit.Loads,'duty');
%     Regulators(ii).kwhdays = get(DSSCircuit.Loads,'kwhdays');
%     Regulators(ii).model = get(DSSCircuit.Loads,'Model');
%     Regulators(ii).numCust = get(DSSCircuit.Loads,'NumCust');
%     Regulators(ii).Rneut = get(DSSCircuit.Loads,'Rneut');
%     Regulators(ii).spectrum = get(DSSCircuit.Loads,'Spectrum');
%     Regulators(ii).VmaxPU = get(DSSCircuit.Loads,'Vmaxpu');
%     Regulators(ii).VminEmerg = get(DSSCircuit.Loads,'Vminemerg');
%     Regulators(ii).VminNorm = get(DSSCircuit.Loads,'Vminnorm');
%     Regulators(ii).VminPU = get(DSSCircuit.Loads,'Vminpu');
%     Regulators(ii).Xneut = get(DSSCircuit.Loads,'Xneut');
%     Regulators(ii).yearly = get(DSSCircuit.Loads,'Yearly');
%     Regulators(ii).status = get(DSSCircuit.Loads,'Status');
%     Regulators(ii).growth = get(DSSCircuit.Loads,'Growth');
    
end


% %% As long as you are not in faultstudy mode, remove all loads that have zero volts on either side (not disabled but are isolated from the circuit)
% if ~isempty(Regulators) && isempty(varargin) && ~strcmp(DSSCircuit.Solution.ModeID,'Faultstudy')
%     condition = [Regulators.voltage]>100;
%     Regulators = Regulators(condition);
% end

catch err
    if ~strcmp(err.identifier,'regName:notfound')
        allLines = [err.stack.line];
        allNames = {err.stack.name};
        fprintf(1, ['\nThere was an error in ' allNames{end} ' in line ' num2str(allLines(end)) ':\n'])
        fprintf(1, ['"' err.message '"' '\n\n'])
        fprintf(1, ['About to run circuitCheck.m to ensure the circuit is set up correctly in OpenDSS.\n\n'])
        fprintf(1, 'If the problem persists, change the MATLAB debug mode by entering in the command window:\n >> dbstop if caught error\n\n')
        fprintf(1, 'Running circuitCheck.m ....................\n')

        warnSt = circuitCheck(DSSCircObj, 'Warnings', 'on')
        assignin('base','warnSt',warnSt);
    end
    rethrow(err);
end
end