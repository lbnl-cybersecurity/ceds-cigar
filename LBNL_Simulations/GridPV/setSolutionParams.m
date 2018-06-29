function  setSolutionParams(DSSObj,mode,number,stepsize,ControMode,varargin)
% The Function parameters are (DSSObj, mode, number, stepsize, ControMode,
% varargin)
% varargin = Max Control Iterations and Iterations, optional arguements
% This function is suitable for QSTS Simulation where a number of
% parameters are set from the main function to run a QSTS simulation. 
% DSSObj is the OpenDSS Object
% mode is a string which can be 'daily' or 'yearly'; not case sensitive
% number = number of solutions executed per solve command, positive integer
% number
% stepsize is the resolution of simulation in second
% ControlMode has to be a string can be 'OFF', 'STATIC', 'EVENT' or 'TIME', not case sensitive
% MaxControlIteration= Maximum number of Control Iterations, default value = 1000
% Iteration= max number of powerflow iterations, defualt value =100 

DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;

if strcmpi(mode,'daily')
    DSSSolution.Mode=1;
elseif strcmpi(mode,'yearly')     
    DSSSolution.Mode=2;
else
    error('Solution Mode not Supported Yet.')
end

if (number > 0)
    DSSSolution.Number=number;
else
    error ('Number of Solutions has to be positive');
end
    
if (stepsize >0)
    DSSSolution.StepSize=stepsize;
else
    error('Stepsize has to be in positive');
end

if strcmpi(ControMode,'OFF')
    DSSSolution.ControlMode=-1;
elseif strcmpi(ControMode,'Static')
    DSSSolution.ControlMode=0;
elseif strcmpi(ControMode,'EVENT')
    DSSSolution.ControlMode=1;
elseif strcmpi(ControMode,'TIME')
    DSSSolution.ControlMode=2;
else
    error ('Control Mode Not Supported by OpenDSS.')
end

numvarargs = length(varargin);
if numvarargs > 3
    error('myfuns:somefun2Alt:TooManyInputs', ...
        'requires at most 3 optional inputs');
end
optargs = {1000 100};
optargs(1:numvarargs) = varargin;
% [DSSSolution.MaxControlIterations,DSSSolution.MaxIterations]=optargs; % Increase the number of maximum control iterations to make sure the system can solve the power flow
% disp(optargs{1}) 
DSSSolution.MaxControlIterations=optargs{1};
DSSSolution.MaxIterations=optargs{2};
end

