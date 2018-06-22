function [DSSObj,boolean] = setCapInfo( DSSObj,capname,property,value,varargin )
% This function allows to set the KVAR, kv, numofstates amd steps info for
% a capacitor. 
% If there are 5 steps for a capacitor, and if the desired condition for
% the steps are 1,1,0,0,0, the parameter should be a number 11000
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
Caps=DSSCircuit.Capacitors;
if (NameChecker==1)
	AllCapNames=Caps.AllNames;
    position=ismember(AllCapNames,capname);
%     disp(position)
	if (sum(position)~=length(capname))
        error('Load Not Found');
    end
end 

if (length(value)~= length(capname))
    error ('Data Input Error, number of Capacitors and number of values do not match')
end
for counter= 1:length(value)
    Caps.Name=capname{counter};
    switch property
        case 'kv'
            Caps.kV=value(counter);
            boolean=1;
        case 'kvar'
             Caps.kvar=value(counter);
             boolean=1;
        case 'numsteps'
            Caps.NumSteps=value(counter);
            boolean=1;
        case 'states'
            Ndigits = dec2base(value(counter),10) - '0';
            feature('COM_SafeArraySingleDim',1);
            Caps.States=Ndigits';
            feature('COM_SafeArraySingleDim',0);
            boolean=1;
        otherwise
            warning ('No Property Matched')
    end

end

