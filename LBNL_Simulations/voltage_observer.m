function [yk,psik,epsilonk] = voltage_observer(vk, vkm1, psikm1, epsilonkm1, ykm1, f_hp, f_lp, gain, T)

    Vmagk = abs(vk);
    Vmagkm1 = abs(vkm1);

    %high pass filter
    
    psik = (Vmagk - Vmagkm1 - (f_hp*T/2-1)*psikm1)/(1+f_hp*T/2);
    
    %square signal
    
    epsilonk = gain*psik^2;
    
    %low pass filter
    
    yk = (T*f_lp*(epsilonk + epsilonkm1) - (T*f_lp - 2)*ykm1)/(2 + T*f_lp);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
