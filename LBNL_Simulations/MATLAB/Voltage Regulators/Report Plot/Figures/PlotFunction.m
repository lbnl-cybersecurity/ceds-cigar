
%
close all
voltwatt=load('voltwatt.mat');
voltvar=load('voltvar.mat');
combimode = load ('Combimode.mat');

time=voltwatt.time;
tap_voltwatt=voltwatt.tap;
tap_voltvar=voltvar.tap;
tap_combimode = combimode.tap;
figure
plot(time,tap_voltwatt,'r',time,tap_voltvar,'b',time,tap_combimode,'k','linewidth',1.5);
legend('VoltWatt Control','Voltvar Control');
xlim([1 length(time)]);
ylabel('Tap Position (pu)')

P_voltwatt=voltwatt.Power;
P_voltvar=voltvar.Power;
P_combimode = combimode.Power;
figure
plot(time,-P_voltwatt,'r',time,-P_voltvar,'b',time,-P_combimode,'k','linewidth',1.5);
legend('VoltWatt Control','Voltvar Control');
xlim([1 length(time)]);
ylabel('Real Power (kW)')

Q_voltwatt=voltwatt.Qvar;
Q_voltvar=voltvar.Qvar;
Q_combimode = combimode.Qvar;
figure
plot(time,-Q_voltwatt,'r',time,-Q_voltvar,'b',time,-Q_combimode,'k','linewidth',1.5);
legend('VoltWatt Control','Voltvar Control');
xlim([1 length(time)]);
ylabel('Rective Power (kVAR)')





