function [qk,pk,gammakused, gammak] = inverter_VoltVarVoltWatt_model(gammakmdelay,...
            solar_irr,Vk,Vkm1,VBP,T,lpf,Sbar,...
    pkm1,qkm1,ROC_lim,InverterRateOfChangeActivate,....
    ksim,Delay_VoltageSampling)

    %VBP = [VQ_start,VQ_end,VP_start,VP_end]

    Vmagk = abs(Vk);
    Vmagkm1 = abs(Vkm1);
    
    %lowpass filter of voltage magnitude
    gammak = (T*lpf*(Vmagk + Vmagkm1) - (T*lpf - 2)*gammakmdelay)/(2 + T*lpf);
    

    if ksim == 1 || mod(ksim, Delay_VoltageSampling(knode)) == 0
        gammakused = gammak     
    else 
        gammakused = gammakmdelay % we don't recalculate it
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %check if solar irradiance is greater than 0
    if (solar_irr < 0.00025)
        pk = 0;
        qk = 0;
    else
        if( gammakused <= VBP(3))
            %no curtailment
            q_avail = (Sbar^2 - solar_irr^2)^(1/2);
            pk = -solar_irr;
        
            % determine VoltVAR support
            if( gammakused <= VBP(1) )
                qk = 0; %no VAR support
            elseif( gammakused > VBP(1) && gammakused <= VBP(2) )
                c = q_avail/(VBP(2) - VBP(1));
                qk = c*(gammakused - VBP(1)); 
                %partial VAR support
            end
        
        elseif( gammakused > VBP(3) && gammakused < VBP(4) )
            %partial curtailment
            d = -1/(VBP(4) - VBP(3));
            pk = d*(gammakused - VBP(3))*solar_irr;
            qk = (Sbar^2 - pk^2)^(1/2);       
        elseif( gammakused >= VBP(4) )
            %full curtailment for VAR support
            qk = Sbar;
            pk = 0;
        end
    
%         ROC limiting
%         pk   
%         if (InverterRateOfChangeActivate==1)
            if(pk - pkm1 > ROC_lim)
                pk = pkm1 + ROC_lim;
            elseif(pk - pkm1 < -ROC_lim)
                pk = pkm1 - ROC_lim;
            end
        
         %qk
            if(qk - qkm1 > ROC_lim)
                qk = qkm1 + ROC_lim;
            elseif(qk - qkm1 < -ROC_lim)
                qk = qkm1 - ROC_lim;
            end    
    end
end
 
