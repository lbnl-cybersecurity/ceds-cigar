function [DSSObj,boolean] = setRegInfo( DSSObj,regname,property,value,varargin )
% This function set the values of properties for a regulator.
%
% [DSSObj,boolean] = setLoadInfo( DSSObj,loadname,property,value,varargin )
% DSSobj= OpenDSS object; 
% regname is the name of the regulators to be changed which can be a cell
% array.
% Property is a string and not case sensitive.
% Value is an array of numbers.
% Length of regname and length of value should match, otherwise an error
% is thrown.
% The last input is an optional input, if the user is unsure about the name
% of the regulator, varargin should be 1 to ensure a sanity check, should be
% avoided if fast operation is required.
% The current implementation does not support multi property setting at the
% same time .
% Current implementation allows to change maxtapchange,debugtrace,eventlog, tapdelay, tapnumber,delay, transformer.
% Changing Transformer Name is not recommended, as it requires to change a
% lot of prperties.
% boolean is set to 1, if property is set properly.
%
% Example:
% setRegInfo(DSSObj,[{'lvr-lvr_01'},{'lvr-lvr_02'}],'MaxTapChange',[2,3]) changes
% the MaxTapChange value of lvr-lvr_01 and lvr-lvr_02 to 2 and 3 respectively.


boolean=0;
NameChecker=0;
AdditionalInputLength=length(varargin);
DSSText = DSSObj.Text;
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
regcontrols=DSSCircuit.RegControls;
if (NameChecker==1)
	AllRegNames=regcontrols.AllNames;
    position=ismember(AllRegNames,regname);
%     disp(position)
	if (sum(position)~=length(regname))
        error('Regulator Not Found');
    end
end 

if (length(value)~= length(regname))
    error ('Data Input Error, number of loads and number of values do not match')
end
for counter= 1:length(value)
    regcontrols.Name=regname{counter};
    switch property
        case 'maxtapchange'
            regcontrols.MaxTapChange=value(counter);
            boolean=1;
        case 'delay'
             regcontrols.Delay=value(counter);
             boolean=1;
        case 'tapdelay'
            regcontrols.TapDelay=value(counter);
            boolean=1;
        case 'tapnumber'
            regcontrols.TapNumber=value(counter);
            boolean=1;
         case 'transformer'
            regcontrols.Transformer=value(counter);
            boolean=1; 
        case 'debugtrace'
            if (value(counter)>0)
                DSSText.command=strcat('RegControl.',regname{counter},'.debugtrace=yes');
            else
                DSSText.command=strcat('RegControl.',regname{counter},'.debugtrace=no');
            end    
            boolean=1; 
         case 'eventlog'
            if (value(counter)>0)
                DSSText.command=strcat('RegControl.',regname{counter},'.eventlog=yes');
            else
                DSSText.command=strcat('RegControl.',regname{counter},'.eventlog=no');
            end    
            boolean=1;    
        case 'enabled'
            DSSCircuit.SetActiveElement(strcat('RegControl.',regname{counter}));
            DSSCircuit.ActiveElement.Enabled=value(counter);
        otherwise
            warning ('No Property Matched')
    end

end

