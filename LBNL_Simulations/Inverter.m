classdef Inverter
   properties  (Access=public)
      Name
      KW
      KVAR
      VBP
      Delay_VBPCurveShift=0
      Delay_VoltageSampling=0
      PercentHacked=0
      ROC_lim=10
      InverterRateOfChangeActivate=0
      TimeStep=1
      LPF
      knode=0
      upk
      uqk
      kp=1
      kq=1
      ThreshHold_vqvp=0.25
      Hacked = 0
      HighPassFilterFrequency = 1
      LowPassFilterFrequency =0.1
      Gain_Energy=1e5
      ksim=0
   end
   methods
%        function obj=Inverter(val)
%             if nargin>0
% 
%             end
%        end
      function [r1,r2] = returnPowerValue(obj)
         r1  = obj.KW;
         r2 = obj.KVAR ;
      end
      function [qk,pk,gammakused,gammakcalc]=voltvarvoltwatt(obj,gammakm1,solar_irr,Vk,Vkm1,VBP,Sbar,pkm1,qkm1,ksim)
        [qk,pk,gammakused, gammakcalc]=inverter_model(gammakm1,...
            solar_irr,Vk,Vkm1,VBP,obj.TimeStep,obj.LPF,Sbar,pkm1,qkm1,...
            obj.ROC_lim,obj.InverterRateOfChangeActivate,...
            ksim,obj.Delay_VoltageSampling,obj.knode);
       end
      
      function uk = adaptivecontrolreal(obj, vk, vkmdelay,ukmdelay, yk)
        uk=adaptive_control(obj.Delay_VBPCurveShift, obj.kp, vk, vkmdelay, ukmdelay, obj.ThreshHold_vqvp,...
                                yk);
        
      end
      
      function uk = adaptivecontrolreactive(obj, vk, vkmdelay,ukmdelay, yk)
        uk=adaptive_control(obj.Delay_VBPCurveShift, obj.kq, vk, vkmdelay, ukmdelay, obj.ThreshHold_vqvp,...
                                yk);
      end
      function [yk,psik,epsilonk] = voltageobserver(obj,vk, vkm1, psikm1, epsilonkm1, ykm1)
        [yk,psik,epsilonk] = voltage_observer(vk, vkm1, psikm1, epsilonkm1, ykm1,...
            obj.HighPassFilterFrequency, obj.LowPassFilterFrequency, obj.Gain_Energy, obj.TimeStep);
      end
   end
end