function [qk,pk,gammak] = inverter_VoltVarVoltWatt_model(gammakm1,...
    solar_irr,Vk,Vkm1,VBP,T,lpf,Sbar,...
    pkm1,qkm1,ROC_lim,InverterRateOfChangeActivate)

%VBP = [VQ_start,VQ_end,VP_start,VP_end]

    %note VQ_end should be equal to VP_start
    %we don't enforce this, but it should be true

    Vmagk = abs(Vk);
    Vmagkm1 = abs(Vkm1);
    
    %lowpass filter of voltage magnitude
    gammak = (T*lpf*(Vmagk + Vmagkm1) - (T*lpf - 2)*gammakm1)/(2 + T*lpf);
    
%     %determine curtailment for VoltWatt
%     if( gammak <= VP_start  )
%         %no curtailment
%         q_avail = (Sbar^2 - solar_irr^2)^(1/2);
%         pk = -solar_irr;
%         
%         % determine VoltVAR support
%         if( gammak <= VQ_start )
%             qk = 0; %no VAR support
%         elseif( gammak > VQ_start && gammak < VQ_end )
%             c = q_avail/(VQ_end - VQ_start);
%             qk = c*(gammak - VQ_start); 
%             %partial VAR support
%         end
%         
%     elseif( gammak > VP_start && gammak < VP_end )
%         %partial curtailment
%         d = -1/(VP_end - VP_start);
%         pk = d*(gammak - VP_start)*solar_irr;
%         qk = (Sbar^2 - pk^2)^(1/2);       
%     elseif( gammak >= VP_end )
%         %full curtailment for VAR support
%         qk = Sbar;
%         pk = 0;
%     end
%     
%     %ROC limiting
%     %pk
%     if(pk - pkm1 > ROC_lim)
%         pk = pkm1 + ROC_lim;
%     elseif(pk - pkm1 < -ROC_lim)
%         pk = pkm1 - ROC_lim;
%     end
%     
%     %qk
%     if(qk - qkm1 > ROC_lim)
%         qk = qkm1 + ROC_lim;
%     elseif(qk - qkm1 < -ROC_lim)
%         qk = qkm1 - ROC_lim;
%     end
%     
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %check if solar irradiance is greater than 0
    if (solar_irr < 0.00025)
        pk = 0;
        qk = 0;
    else
        if( gammak <= VBP(3))
            %no curtailment
            q_avail = (Sbar^2 - solar_irr^2)^(1/2);
            pk = -solar_irr;
        
            % determine VoltVAR support
            if( gammak <= VBP(1) )
                qk = 0; %no VAR support
            elseif( gammak > VBP(1) && gammak <= VBP(2) )
                c = q_avail/(VBP(2) - VBP(1));
                qk = c*(gammak - VBP(1)); 
                %partial VAR support
            end
        
        elseif( gammak > VBP(3) && gammak < VBP(4) )
            %partial curtailment
            d = -1/(VBP(4) - VBP(3));
            pk = d*(gammak - VBP(3))*solar_irr;
            qk = (Sbar^2 - pk^2)^(1/2);       
        elseif( gammak >= VBP(4) )
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
%         end
       
    end
    
end
