function [DSSObj,boolean] = setLoadInfo( DSSObj,loadname,property,value,varargin )

% This function set the values of properties for a load.
%
% [DSSObj,boolean] = setLoadInfo( DSSObj,loadname,property,value,varargin )
% DSSobj= OpenDSS object; 
% loadname is the name of the loads to be changed which can be a cell
% array.
% Property is a string and not case sensitive.
% Value is an array of numbers.
% Length of loadname and length of value should match, otherwise an error
% is thrown.
% The last input is an optional input, if the user is unsure about the name
% of the load, varargin should be 1 to ensure a sanity check, should be
% avoided if fast operation is required
% The current implementation does not support multi property setting at the
% same time .
% Current implementation allows to change KW, KVAR, pf and KVA.
% boolean is set to 1, if property is set properly.
%
% Example:
% setLoadInfo(DSSObj,[{'load_645'},{'load_611'}],'kw',[200,300]) changes
% the KW value of load_645 and load_611 to 200 and 300 respectively.


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
Loads=DSSCircuit.Loads;
if (NameChecker==1)
	AllLoadNames=Loads.AllNames;
    position=ismember(AllLoadNames,loadname);
%     disp(position)
	if (sum(position)~=length(loadname))
        error('Load Not Found');
    end
end 

if (length(value)~= length(loadname))
    error ('Data Input Error, number of loads and number of values do not match')
end
for counter= 1:length(value)
    Loads.Name=loadname{counter};
    switch property
        case 'kw'
            Loads.kW=value(counter);
            boolean=1;
        case 'kvar'
             Loads.kvar=value(counter);
             boolean=1;
        case 'kva'
            Loads.kva=value(counter);
            boolean=1;
        case 'pf'
            Loads.PF=value(counter);
            boolean=1;
        case 'model'
            Loads.Model=value(counter);
            boolean=1;
        otherwise
            warning ('No Property Matched')
    end

end

