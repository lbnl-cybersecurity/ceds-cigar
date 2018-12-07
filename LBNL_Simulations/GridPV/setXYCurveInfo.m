function [DSSObj,boolean] = setXYCurveInfo( DSSObj,XYCurvenames,value,varargin )
% The value is a structure with three properties:
% npts = number of points
% xarray = corresponding x array values, which should be of dimension
% 1xnpts
% yarray = corresponding y array values which should be of dimension
% 1xnpts
% For each xycurve, a structure has to be sent from the main function 

DSSCircuit = DSSObj.ActiveCircuit;
XYCurves=DSSCircuit.XYCurves;
boolean=0;
NameChecker=0;
if (XYCurves.Count==0)
    error('No XYCurve Found in the OpenDSS Model.')
end

if (sum(class(value)=='struct')==0)
    error('Structure Expected as Input Value.')
end
AdditionalInputLength=length(varargin);
switch AdditionalInputLength
	case 0
		NameChecker=0;
	case 1
		NameChecker=(varargin{1});
	otherwise
end

if (NameChecker==1)
       AllXYCurveNames = cell(1,XYCurves.Count);
       XYCurves.First;
       for ii = 1:XYCurves.Count
            AllXYCurveNames{ii} = XYCurves.Name;
            XYCurves.Next;
       end
       position=ismember(AllXYCurveNames,XYCurveNames);
       if (sum(position)~=length(XYCurveNames))
        error('XYCurve Not Found.');
       end
end


if (length(value) ~= length(XYCurvenames))
    error('Number of XYCurvenames and Number of property do not match. ');
    
end

for i=1:length(XYCurvenames)
    if (value(i).npts ~= length(value(i).xarray) || value(i).npts ~= length(value(i).yarray)) 
        error('Input Value Mismatch.');
    end
    if (numel(fieldnames(value(i))) ~=4)
        error('You Need to Set All the properties.');
    end
    XYCurves.Name=XYCurvenames{i};
    XYCurves.Npts=value(i).npts;
    feature('COM_SafeArraySingleDim',1);
    XYCurves.Xarray=value(i).xarray';
    feature('COM_SafeArraySingleDim',0);
    feature('COM_SafeArraySingleDim',1);
    XYCurves.Yarray=value(i).yarray';
    feature('COM_SafeArraySingleDim',0);
    boolean=1;
end