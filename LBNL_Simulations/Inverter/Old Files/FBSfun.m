function [V,I,S,iter] = FBSfun(V0,loads,Z,B)
%note: loads = (p+j*q)*[a_S,a_I,a_Z]

n = length(Z);
%setup voltages and currents
V = zeros(n,1);
s = V;
V(1) = V0;
I = zeros(n,1);
I(1) = 0;
Vp=V;

%sort nodes into terminal (T) set and junction (J) set
%T: nodes which only have -1
%J: nodes which have more than one 1
T = [];
J = [1];
for k=2:1:n
    %loop through rows
    t = sum(B(k,:));
    if(t == -1)
        %node is terminal node
        T = [T,k];
    elseif(t >= 1)
        %junction node
        J = [J,k];
    end
end

tol = 0.0001;
iter = 0;
Vtest = 0;

while(abs(Vtest-V0) >= tol)
    V(1) = V0;
    Vp(1) = V0;
    %sweep forward, calculate voltages at nodes
    for k=1:1:n-1
        [abs_val,idx,val] = find(B(k,:)>0);
        V(idx)=V(k)-Z(idx).*I(idx);
        Vp(idx)=V(k)-Z(idx).*I(idx);
        %loop through nonzero entries, look for child node
%         for t=1:1:length(val)
%             %look for a child node
%             if(val(t) == 1)
%                 %we've found a child, calculate voltage at child
%                 V(idx(t)) = V(k) - Z(idx(t))*I(idx(t));
%             end
%         end
    end
    
    %sweep backward
    clear t
    %TERMINAL NODES
    %loop through terminal nodes
    for k=length(T):-1:1
        t = T(k);
        %evaluate ZIP loads for present voltage and current
        s(t) = loads(t,:)*[1,abs(V(t)),abs(V(t))^2]';
        %calculate segment current
        I(t) = conj(s(t)/V(t));
        %move up the feeder until a junction node is hit
        flag = true;
        [abs_val,idx,val] = find(B(t,:) == -1);
        while(flag)
            %assume each node only has 1 parent
            %calculate voltage at parent node
            V(idx) = V(t) + Z(t)*I(t);
            %calculate current in parent segment
            s(idx) = loads(idx,:)*[1,abs(V(idx)),abs(V(idx))^2]';
            I(idx) = conj(s(idx)/V(idx)) + I(t);
            %check if idx is a junction node
            if(isempty(find(J == idx)))
                %then this is not a junction node
                
                %update t, idx and proceed upstream to next node
                t = idx;
                [abs_val,idx,val] = find(B(idx,:) == -1);
            else
                %this is a junction node and we stop this branch
                flag = false;
            end
        end
        %completed branches from terminal nodes to junction nodes
        %junction nodes are now terminal branches
    end
    
    %JUNCTION NODES
    %start from bottom-most junction node and repeat until a junction
    %node is hit, stop if junction node is 1
    %loop through junction nodes
    for k=length(J):-1:2
        t = J(k);
        %calculate segment current
        %for all nodes except 1, more than 1 segment is connected
        s(t) = loads(t,:)*[1,abs(V(t)),abs(V(t))^2]';
        load_current = conj(s(t)/V(t));
        [abs_val,idx,val] = find(B(t,:));
        Itot = load_current;
        for y=1:1:length(val)
            %look for a child node
            if(val(y) == 1)
                Itot = Itot + I(idx(y));
            end
        end
        I(t) = Itot;
        %move up the feeder until a junction node is hit
        flag = true;
        [abs_val,idx,val] = find(B(t,:) == -1);
        while(flag)
            %assume each node only has 1 parent
            %calculate voltage at parent node
            V(idx) = V(t) + Z(t)*I(t);
            s(idx) = loads(idx,:)*[1,abs(V(idx)),abs(V(idx))^2]';            
            %calculate current in parent segment
            I(idx) = conj(s(idx)/V(idx)) + I(t);

            %check if idx is a junction node
            if(isempty(find(J == idx)))
                %then this is not a junction node
                %update t, idx and proceed upstream
                t = idx;
                [abs_val,idx,val] = find(B(idx,:) == -1);
            else
                %this is a junction node and we stop this branch
                flag = false;
            end
        end
        %completed branches from terminal nodes to junction nodes
        %junction nodes are now terminal branches
    end
    Vtest = V(1);
    iter = iter+1;
end
V(1) = V0;

%power delivered to each node
S = ones(n,1);
for k=1:1:n
    S(k) = V(k)*I(k)';
end

end
