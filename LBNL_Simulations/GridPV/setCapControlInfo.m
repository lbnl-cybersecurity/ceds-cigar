function [DSSObj,boolean] = setCapControlInfo( DSSObj,capcontrolname,property,value,varargin )
% The function here allows to setup the information for capacitor controls.
% The parameter which can be set through these: delay, delayoff, enabled,
% on setting, offsetting, ptratio, ctratio, terminal, vmax,vmin
boolean=0;
NameChecker=0;
AdditionalInputLength=length(varargin);
%Loweing the case to avoid checking all possible combinations
try
    property=lower(property);
catch
    error('String Expected')
end
switch AdditionalInputLength
	case 0
		NameChecker=0;
	case 1
		NameChecker=(varargin{1});
	otherwise
end	
DSSCircuit = DSSObj.ActiveCircuit;
CapControls=DSSCircuit.CapControls;
if (NameChecker==1)
	AllCapControlNames=CapControls.AllNames;
    position=ismember(AllCapControlNames,capcontrolname);
%     disp(position)
	if (sum(position)~=length(capcontrolname))
        error('Load Not Found');
    end
end 

if (length(value)~= length(capcontrolname))
    error ('Data Input Error, number of Capacitors and number of values do not match')
end
for counter= 1:length(value)
    CapControls.Name=capcontrolname{counter};
    switch property
        case 'delay'
            CapControls.Delay=value(counter);
            boolean=1;
        case 'delayoff'
             CapControls.DelayOFF=value(counter);
             boolean=1;
        case 'offsetting'
            CapControls.OFFsetting=value(counter);
            boolean=1;
        case 'onsetting'
            CapControls.Onsetting=value(counter);
            boolean=1;    
        case 'element'
            CapControls.element=value(counter);
            boolean=1;
        case 'terminal'
            CapControls.terminal=value(counter);
            boolean=1;
        case 'vmax'
            CapControls.Vmax=value(counter);
            boolean=1; 
         case 'vmin'
            CapControls.Vmin=value(counter);
            boolean=1; 
         case 'ptratio'
            CapControls.PTratio=value(counter);
            boolean=1;
         case 'ctratio'
            CapControls.CTratio=value(counter);
            boolean=1  ;
        case 'enabled'
            DSSCircuit.SetActiveElement(strcat('CapControl.',capcontrolname{counter}));
            DSSCircuit.ActiveElement.Enabled=value(counter);
        otherwise
            warning ('No Property Matched')
    end

end

