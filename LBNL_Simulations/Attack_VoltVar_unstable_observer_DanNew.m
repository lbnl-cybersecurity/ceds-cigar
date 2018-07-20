% Daniel Arnold
% Code showing instability of VoltVar VoltWatt control

%05/25/2017 - adapted from EPIC 303 code
%06/14/2017 - added max rate of change (ROC) limiting for p and q output
%08/21/2017 - added code for observer
%09/25/2017 - modified code and plotting functions for PSCC paper

clc, clear all, close all

%% Feeder setup

display('Map feeder')

fdr = 13; % feeder name and also number of nodes in feeder

% obtain Feeder map matrix - FM
% lengths between nodes - SL
% all paths from root to tail node - paths
% list of node names - nodelist
[FM, Z, paths, nodelist, loadlist] = ieee_feeder_mapper(fdr);

display('Set up feeder topography')

% setup voltages and currents
% Vbase = 4.16e3; %4.16 kV
% Sbase = 500e3; %500 kVA

Vbase = 4.16e3; %4.16 kV
Sbase = 500e3; %500 kVA

Zbase = Vbase^2/Sbase;
Ibase = Sbase/Vbase;

V = zeros(fdr,1);
I = zeros(fdr,1);
I(1) = 0;

Z = Z/Zbase*2;
Z(2) = Z(2) + 0.01 + 0.08*j; %adds the transformer impedance to the first line segment impedance
%% Load node data

display('Load node data and insolation data')
%loads at the nodes
pvnum = 10;

fp = [pwd,'/testpvnum' num2str(pvnum) '/'];
%fp = ['D:\pecan\pecan_street_profiles\nodes\testpvnum' num2str(pvnum) '\'];

%only put load data where it appears in the IEEE test feeder descritpion
DATA = zeros(1441,4,fdr);
for node = 1:1:length(loadlist)
    
    fn = ['node_' num2str(node) '_pv_' num2str(pvnum) '_minute.mat'];
    
    S = load([fp fn],'-mat','nodedata');    
    DATA(:,:,loadlist(node)) = S.nodedata;
    
end
time = 0:1:1440;

%% Seperate PV Generation Data
display('Creating PV generation data')

%sort data into gen, grid, use
gen = DATA(:,2,:);
grid = DATA(:,3,:);
use = DATA(:,4,:);

clear fn S

%% Setup Nodal Demands and Generation


display('Convert demand magnitudes to real and reactive components');
display('Setup ZIP loads')
a_S = 0.75;  %constant power
a_I = 0.05; %constant current
a_Z = 0.2; %constant impedance
ZIP_demand = zeros(length(time),fdr,3);
ZIP_demand_as = ZIP_demand;
%ZIP loads for real and react pow

%scale demands for simulation
load_scale = 100/Sbase;
gen = gen*load_scale*6;
grid = grid*load_scale;
use = use*load_scale*0.9*6;

gen_scale = 2;  %multiplier to increase amount of renewable gen

gen_max = zeros(fdr,1);
temp = max(gen);
for k=1:fdr
    gen_max(k) = temp(1,1,k);
end

% convert magnitudes to real and reactive power demand
for node = 2:fdr
    %apply 0.9 power factor to electricity use data and make loads ZIP
    ZIP_demand(:,node,:) = [a_S*use(:,1,node), a_I*use(:,1,node), ...
    a_Z*use(:,1,node)]*(0.9 + 1i*0.4359)/0.9;
end

 
%% Compute V0 (time varying)
% V0 = 1.0 + gen(:,1,5)/max( gen(:,1,5) ) * 0.01;
V0 = 1.02*ones(size(gen(:,1,5)));


%% Simulation

display('Set up simulation')

%global parameters
startTime = time(1);
stopTime = time(end);
l = length(time);

%Initialize feeder states
V_nc = zeros(fdr,l);
S_nc = V_nc;

V_vqvp = V_nc;
S_vqvp = S_nc;

%setup inverter
% [qk,pk,gammak] = inverter_VoltVarVoltWatt_model(gammakm1,...
%     solar_irr,Vk,Vkm1,VQ_start,VQ_end,VP_start,VP_end,T,lpf)
qk = zeros(size(gen));
pk = qk;
ROC_lim = 0.01; %rate of change limit
% ROC_lim = 0.1;
gamma = qk;
T = 1;
lpf = 1;

V1 = 1.01;
V2 = 1.012;
V3 = 1.012;
V4 = 1.02;

V1 = 1.01;
V2 = 1.03;
V3 = 1.03;
V4 = 1.05;

nc = length(loadlist);

VBP = [V1*ones(fdr,1), V2*ones(fdr,1), ...
    V3*ones(fdr,1), V4*ones(fdr,1)];

Sbar = zeros(fdr,1);

%setup observer
%[yk,psik,epsilonk] = voltage_observer(vk, vkm1, f_hp, f_lp, gain, T)
yk_nc = zeros(length(time),length(nodelist));
yk = yk_nc;
psik_nc = yk;
psik = yk;
epsilonk_nc = yk;
epsilonk = yk;
f_lp = 0.1;
f_hp = 1;
gain = 1e5;

thresh = 0.1; %observer threshold for oscillation detection

%setup adaptive controller
upk = zeros(size(psik));
uqk = upk;
kq = 100;
kp = 100;

%generation values
gen_val_bl = zeros(l,fdr);
gen_val_vqvp = gen_val_bl;

%setup inverter capacities - these are static parameters
% Sbar(:) =  1.15 * gen_max * gen_scale;
Sbar(:) =  1 * gen_max * gen_scale;
    
%simulation
for ksim = 1:1:length(time)
    
    %No Control Case --------------------------------------------------
    %this case is full feed-through of active power generation
    
    %setup loads for this timestep
    sbl = zeros(fdr,3);
    for node_iter = 1:numel(loadlist)
        knode = loadlist(node_iter);
            
        gen_val_bl(ksim,knode) = gen(ksim,1,knode) * gen_scale;
            if(gen_val_bl(ksim,knode) > Sbar(knode))
                gen_val_bl(ksim,knode) = Sbar(knode);
            end
            
        sbl(knode,:) = [ZIP_demand(ksim,knode,1) - gen_val_bl(ksim,knode),...
            ZIP_demand(ksim,knode,2),ZIP_demand(ksim,knode,3)];

    end
    
    [Vrun,Irun,Srun,iter] = FBSfun(V0(ksim),sbl,Z,FM);
    V_nc(:,ksim) = Vrun;
    S_nc(:,ksim) = Srun;
    iters_nc(ksim) = iter;
    
    %observer
    for knode=1:1:numel(nodelist)
        if(ksim > 1)
            [yk_nc(ksim,knode),psik_nc(ksim,knode),epsilonk_nc(ksim,knode)] = voltage_observer(V_nc(knode,ksim), ...
                V_nc(knode,ksim-1), psik_nc(ksim-1,knode), ...
                epsilonk_nc(ksim-1,knode), yk_nc(ksim-1,knode), f_hp, f_lp, gain, T);
        end
    end
    
    %VQVP Control Case ------------------------------------------------
    
    %setup loads for this timestep
    s_vqvp = zeros(fdr,3);
    for node_iter = 1:numel(loadlist)
        knode = loadlist(node_iter);
        if(ksim > 1)
                
            gen_val_vqvp(ksim,knode) = gen(ksim,1,knode) * gen_scale;
                
            s_vqvp(knode,:) = [ZIP_demand(ksim,knode,1) + pk(ksim-1,1,knode) + ...
                2*1i*qk(ksim-1,1,knode), ZIP_demand(ksim,knode,2), ...
                ZIP_demand(ksim,knode,3)];
        else
            s_vqvp(knode,:) = [ZIP_demand(ksim,knode,1) - gen_val_vqvp(ksim,knode), ...
                ZIP_demand(ksim,knode,2),ZIP_demand(ksim,knode,3)];
        end
    end
    
    [Vrun,Irun,Srun,iter] = FBSfun(V0(ksim),s_vqvp,Z,FM);
    V_vqvp(:,ksim) = Vrun;
    S_vqvp(:,ksim) = Srun;
    iters_vqvp(ksim) = iter;
    
    %compute inverter action
    if(ksim > 1 && ksim < length(time))
        for node_iter = 1:numel(loadlist)
        knode = loadlist(node_iter);
            [qk(ksim,1,knode),pk(ksim,1,knode),gamma(ksim,1,knode)] = ...
                inverter_VoltVarVoltWatt_model(gamma(ksim-1,1,knode),...
                gen_val_vqvp(ksim,knode),...
                abs(V_vqvp(knode,ksim)),abs(V_vqvp(knode,ksim-1)),...
                VBP(knode,:),T,lpf,Sbar(knode),...
                qk(ksim-1,1,knode),pk(ksim-1,1,knode),ROC_lim);
        end
    end
    
    %observer and adaptive control
    for node_iter=1:1:numel(loadlist)
        knode = loadlist(node_iter);
        
        if (ksim>1)
            [yk(ksim,knode),psik(ksim,knode),epsilonk(ksim,knode)] = voltage_observer(V_vqvp(knode,ksim), ...
                V_vqvp(knode,ksim-1), psik(ksim-1,knode), ...
                epsilonk(ksim-1,knode), yk(ksim-1,knode), f_hp, f_lp, gain, T);
        end
        
        %check to see if we're seeing unstable oscillations
        if(ksim > 1 && ksim < length(time))
            if( ismember(knode,loadlist) == 1)
                %this node is controllable (has DER)
                idx = find(loadlist == knode);
                %re-set VBP
                upk(ksim,knode) = adaptive_control(T, kp, psik(ksim,knode), ...
                    psik(ksim-1,knode), upk(ksim-1,knode), thresh, yk(ksim,knode));
                uqk(ksim,knode) = adaptive_control(T, kq, psik(ksim,knode), ...
                    psik(ksim-1,knode), uqk(ksim-1,knode), thresh, yk(ksim,knode));
 
                VBP(knode,:) = [V1-uqk(ksim,knode),V2-uqk(ksim,knode)/2,V3+upk(ksim,knode)/2,V4+upk(ksim,knode)];
%                 VQ_start = 1.01; VQ_end = 1.011; VP_start = 1.011; VP_end = 1.03;
            end
        end
    
    end
    
end
    
display( 'Simulation Complete' )

%% Calculate PV penetration

%calculate max noncoincident demand
load_sum = zeros(length(time),1);
for k1= 1:length(time)
    load_sum(k1) = sum(ZIP_demand(k1,:,1));
end
load_max = max(abs(load_sum));

%installed capacity
max_pv = sum(Sbar(:));
pv_pen = max_pv/load_max;

display(['ratio of max PV output to feeder peak load: ', num2str(pv_pen)] )

%%  Calculate feeder head powers
P0_nc = real(S_nc(1,:));
Q0_nc = imag(S_nc(1,:));

P0_vqvp = real(S_vqvp(1,:));
Q0_vqvp = imag(S_vqvp(1,:));

%% Figures - overlay
close all
t1 = datetime(2017,8,0,0,0,0);
t_datetime = t1 + minutes(time);
 
% legend({'Base case','ES','Max real pow','Min real pow','Target'},'FontSize',14,'FontName','Times New Roman','Location','northwest','Orientation','vertical')
% % legend('boxoff')
% xlabel('Time','FontSize',20,'FontName','Times New Roman')
% ylabel('p.u.','FontSize',20,'FontName','Times New Roman')
% title('Feeder Head Real Power','FontSize',20,'FontName','Times New Roman')
% set(gca,'FontSize',16,'FontName','Times New Roman', ...
%     'XTick',datenum(2014,7,1,0:0.25:24,0,0),'XTickLabel',datestr(datenum(2014,7,1,0:0.25:24,0,0),15))

node = 4;
f1 = figure(1);
set(f1,'Units','Inches');
pos = get(f1,'Position');
set(f1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime , abs(V_nc(node,:)), 'b','LineWidth',1.5)
hold on
plot(t_datetime , abs(V_vqvp(node,:)), 'r','LineWidth',1.5)
hold off
% datetick('x','HH:MM')
datetick('x','HH:MM')
title(['Voltage Magnitude, node: ', num2str(nodelist(node))],'FontSize',14)
legend({'feedthrough','VV-VW control'},'FontSize',14);
xlabel('time (hours)')
ylabel('volts (pu)')
set(gca,'FontSize',14,'FontName','Times New Roman')
% set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
%     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
%     datestr(datenum(2017,8,0,0:4:24,0,0),15))

f2 = figure(2);
set(f2,'Units','Inches');
pos = get(f2,'Position');
set(f2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime,yk_nc(:,node), 'b','LineWidth',1.5)
hold on
plot(t_datetime, yk(:,node), 'r','LineWidth',1.5 )
hold off
datetick('x','HH:MM')
title(['Observer Output, node: ', num2str(nodelist(node))])
legend({'feedthrough','VV-VW control'},'FontSize',14);
xlabel('time (hours)')
set(gca,'FontSize',14,'FontName','Times New Roman')
% set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
%     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
%     datestr(datenum(2017,8,0,0:4:24,0,0),15))
ylabel('instability (V^2) (pu)')

f3 = figure(3);
set(f3,'Units','Inches');
pos = get(f3,'Position');
set(f3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime,P0_nc, 'b+','LineWidth',1.5)
hold on
plot(t_datetime,P0_vqvp, 'r+','LineWidth',1.5)
hold off
datetick('x','HH:MM')
legend({'feedthrough','VV-VW control'},'FontSize',14,'Location','SouthEast');
set(gca,'FontSize',14,'FontName','Times New Roman')
% set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
%     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
%     datestr(datenum(2017,8,0,0:4:24,0,0),15))
title('Substation Real Power')
xlabel('time (hours)')
ylabel('Watts (pu)')

f4 = figure(4);
set(f4,'Units','Inches');
pos = get(f4,'Position');
set(f4,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime,Q0_nc, 'b','LineWidth',1.5)
hold on
plot(t_datetime,Q0_vqvp, 'r','LineWidth',1.5)
hold off
datetick('x','HH:MM')
title('Substation Reactive Power')
xlabel('time (hours)')
ylabel('VARs (pu)')
legend({'feedthrough','VV-VW control'},'FontSize',14);
set(gca,'FontSize',14,'FontName','Times New Roman')
% set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
%     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
%     datestr(datenum(2017,8,0,0:4:24,0,0),15))

f5 = figure(5);
set(f5,'Units','Inches');
pos = get(f5,'Position');
set(f5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime,pk(:,1,4), 'b','LineWidth',1.5)
hold on
plot(t_datetime,qk(:,1,4), 'r','LineWidth',1.5)
hold off
datetick('x','HH:MM')
title(['DER Power Output, node: ', num2str(nodelist(4))])
xlabel('time (hours)')
ylabel('pu')
legend({'active power','reactive power'},'FontSize',14);
set(gca,'FontSize',14,'FontName','Times New Roman')

f6 = figure(6);
set(f6,'Units','Inches');
pos = get(f6,'Position');
set(f6,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[ceil(pos(3)), ceil(pos(4))])
plot(t_datetime,upk(:,6), 'b','LineWidth',1.5)
hold on
plot(t_datetime,uqk(:,6), 'r','LineWidth',1.5)
hold off
datetick('x','HH:MM')
title(['up and uq: ', num2str(nodelist(4))])
xlabel('time (hours)')
ylabel('pu')
legend({'active power','reactive power'},'FontSize',14);
set(gca,'FontSize',14,'FontName','Times New Roman')

% set(gca,'FontSize',14,'FontName','Times New Roman','XTick',...
%     datenum(2017,8,0,0:4:24,0,0),'XTickLabel',...
%     datestr(datenum(2017,8,0,0:4:24,0,0),15))

% %% Figures - single
% t1 = datetime(2017,8,0,0,0,0);
% t_datetime = t1 + minutes(time);
% 
% node = 2;
% f1 = figure(1);
% plot(t_datetime,abs(V_vqvp(node,:)), 'b' )
% datetick('x','HH:MM')
% title(['Voltage Magnitude, node: ', num2str(nodelist(node)), ' (control)'])
% xlabel('time (hours)')
% ylabel('volts (pu)')
% 
% f2 = figure(2);
% plot(t_datetime,abs(V_nc(node,:)), 'b' )
% datetick('x','HH:MM')
% title(['Voltage Magnitude, node: ', num2str(nodelist(node)), ' (no control)'])
% xlabel('time (hours)')
% ylabel('volts (pu)')
% 
% f3 = figure(3);
% plot(t_datetime,yk_nc(:,node), 'r' )
% datetick('x','HH:MM')
% title(['Observer Output (no control), node: ', num2str(nodelist(node))])
% xlabel('time (hours)')
% ylabel('instability (V^2) (pu)')
% 
% f4 = figure(4);
% plot(t_datetime,yk(:,node), 'r' )
% datetick('x','HH:MM')
% title(['Observer Output (control), node: ', num2str(nodelist(node))])
% xlabel('time (hours)')
% ylabel('instability (V^2) (pu)')
% 
% f5 = figure(5);
% plot(t_datetime,P0_nc, 'b' )
% datetick('x','HH:MM')
% title('Substation Real Power - Feedthrough')
% xlabel('time (hours)')
% ylabel('Watts (pu)')
% 
% f6 = figure(6);
% plot(t_datetime,Q0_nc, 'b' )
% datetick('x','HH:MM')
% title('Substation Reactive Power - Feedthrough')
% xlabel('time (hours)')
% ylabel('VARs (pu)')
% 
% f7 = figure(7);
% plot(t_datetime,P0_vqvp, 'b' )
% datetick('x','HH:MM')
% title('Substation Real Power - VQVP')
% xlabel('time (hours)')
% ylabel('Watts (pu)')
% 
% f8 = figure(8);
% plot(t_datetime,Q0_vqvp, 'b' )
% datetick('x','HH:MM')
% title('Substation Reactive Power - VQVP')
% xlabel('time (hours)')
% ylabel('VARs (pu)')

%% save the figures

% print(f1,'voltage_control','-dpdf')
% print(f2,'observer_control','-dpdf')
% print(f3,'realpow_control','-dpdf')
% print(f4,'reactpow_control','-dpdf')

% print(f1,'voltage_control','-dpng')
% print(f2,'observer_control','-dpng')
% print(f3,'realpow_control','-dpng')
% print(f4,'reactpow_control','-dpng')

% print(f1,'voltage','-dpdf')
% print(f2,'observer','-dpdf')
% print(f3,'realpow','-dpdf')
% print(f4,'reactpow','-dpdf')