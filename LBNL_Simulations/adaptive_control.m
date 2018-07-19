function [uk] = adaptive_control(delay, k, vk, vkmdelay, ukmdelay, thresh, yk)
% Adaptive controller for re-tuning volt var and volt watt settings

% integrator with squared inputs
if (yk > thresh)
    uk = delay/2*k * ( vk^2 + vkmdelay^2 ) + ukmdelay;
else
    uk = ukmdelay;
end

