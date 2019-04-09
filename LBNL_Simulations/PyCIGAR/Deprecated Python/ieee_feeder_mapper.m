function [FM, Z, paths, nodelist, loadlist] = ieee_feeder_mapper(feeder)

% Read from csv file, input first 4 columns as strings (general case)
% Change file path as needed
fp = 'C:\feeders\feeder13\csv\';
% fp = 'C:\Users\Michael\Desktop\ARPAE\feeder mapping\';
fn = 'Line Data.csv';

data_numeric = csvread([fp fn]);
A = data_numeric(:,1);
B = data_numeric(:,2);
L = data_numeric(:,3);
CONF = data_numeric(:,4);


%% Find root node and tail nodes

for m=1:length(A)
    % loop through rows of A, find entry in A is not found in B. rn
    % will be the row of [A B] with the root node in A.
    if isempty(strmatch(A(m),B,'exact')) == 1
        rn = m;
        break;
    end
end
% rn % index of root node
RN = A(m); % root node names

tn = [];
for m=1:length(B)
    % loop through rows of B, find where entries in B are not found in A.
    % tn will be the rows of [A B] with tail nodes in B. this will only
    % work properly on if nodes only have one (1) parent
    if isempty(strmatch(B(m),A,'exact')) == 1
        tn(end+1) = m;
    end
end
% tn % indeces of tail nodes
TN = B(tn); % tail node names

%% Assign numbers to nodes

% empty list of nodes
nodelist = [RN];
% loop through all tail nodes found
for k1 = 1:length(tn)
    
    nodePathToRoot = [B(tn(k1))]; % path to root node from tail, including tail

    rootfound = false;
    % continue upstream until root node reached
    while(rootfound == false)
        
        % find row of [A B] in which the current node is in, only works if
        % nodes only have a single parent
        k2 = strmatch(nodePathToRoot(end),B,'exact');
        nodePathToRoot(end+1) = A(k2); % add current node to path
        % check if current node is not found in B. if false, continue
        % upstream. if true, root node reached
        if isempty(strmatch(nodePathToRoot(end),B,'exact')) == 1
            rootfound = true;
        end
    end
    
    nodePathToTail = fliplr(nodePathToRoot); % flip path to root to obtain path to tail
    % iterate down from root to tail in current path, checking if the nodes
    % in the path have been listed yet. if listed, ignore. if not listed,
    % add to end of list.
    for k3 = 2:length(nodePathToTail)
        % check if the node in the path from root to tail has not been
        % listed. add to list if not.
        if isempty(strmatch(nodePathToTail(k3),nodelist,'exact')) == 1
            nodelist = [nodelist; nodePathToTail(k3)];
%             nodelist(end+1) = nodePathToTail(k3);
        end
    end
end
% clear currNode nodePathToRoot nodePathToTail rootfound

%%

SL = zeros(length(nodelist),1); % segment lengths
FM = zeros(length(nodelist),length(nodelist)); % matrix of node relationships
% iterate down rows of [A B] (node relationships) and add relationships to
% M, and segment lengths to S
for m=1:length(A)
    % upstream (nA) and downstream (nB) node for relationship
    nA = A(m); nB = B(m);
    % find where in nodelist nA and nB are
    k1 = strmatch(A(m),nodelist,'exact');
    k2 = strmatch(B(m),nodelist,'exact');
    
    FM(k1,k2) = 1; FM(k2,k1) = -1; % add relationship to M    
    SL(k2) = str2double(L(m)); % add segment length to S
    
end

% matrix of all paths from feeder head to tail
% paths that are shorter than the max have zeros added onto them
% use find(k,:) function to obtain kth path

% need to find vector of tail nodes now that we've labeled, this is NOT tn.
%  tn is only a vector of indices.  we will overwrite tn for convenience

tn = [];
for k1=1:length(TN)
    tn(k1) = find(nodelist == TN(k1));
end

paths = [];
for k1 = 1:1:length(tn)
        temppath = tn(k1);
    while temppath(end) ~= 1
        temppath(end+1) = find(FM(temppath(end),:) == -1);
    end
    temppath = fliplr(temppath);
    if length(temppath) > size(paths,2)
        paths = [paths zeros(size(paths,1), length(temppath)-size(paths,2)); temppath];
    elseif length(temppath) == size(paths,2)
        paths = [paths; temppath];
    elseif length(temppath) < size(paths,2)
        paths = [paths; temppath zeros(1,size(paths,2)-length(temppath))];
    end
end

%% Assign impedances to line segments
%create segment_list, should have length one less than nodelist
% impedance convention: line segment immediatley upstream of node has the
% same number, this implies that Z(1) = 0, as node 1 is th source node

%re-order CONF to correspond to our impedance convention
Z_conf = [];
L_ordered = [];
for k1=1:1:length(nodelist)-1
    idx = find(nodelist == B(k1));
    Z_conf(idx,1) = CONF(k1);
    L_ordered(idx,1) = L(k1);
end

%loop through L and replace 0 with something small
for k1=1:1:length(nodelist)
    if(L_ordered(k1) == 0)
        L_ordered(k1) = 1e-1;
    end
end
L_ordered = L_ordered/5280;
%load balanced_line_config file to assign correct impedances.  Impedance
%values are in ohms per mile

%At this time: we're ignoring all switches, line regulators, and
%transformers: we will treat as neglidgably small impedances

Z_per_mile = zeros(feeder,1);
data = csvread([fp 'balanced_line_config.csv']);
for k1=2:1:length(nodelist)
    conf_val = Z_conf(k1);
    if(isnan(conf_val))
        %then I have a transformer, switch or regulator here, so assign a
        %custom impedance
        Z_per_mile(k1) = 0.1+0.1i;
    else
        idx = find(data == conf_val);
        Z_per_mile(k1) = data(idx,2) + 1i*data(idx,3);
    end
end
%multiply by line segment length
Z = Z_per_mile.*L_ordered;

%% Determine Load Vector
%tells us where to place the loads in the circuit
load_vec = csvread([fp 'load_list.csv']);

% nodelist contains the proper ordering of nodes
loadlist = [];
for k1=1:1:length(load_vec)
    idx = find(nodelist == load_vec(k1));
    loadlist = [loadlist idx];
end

end