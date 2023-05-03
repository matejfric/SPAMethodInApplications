function S = myspa_compute_S(Gamma,X)
%SPA_COMPUTE_S Calculation of the SPA cluster parameters

K = size(Gamma,1);

if true
    S = ((Gamma*Gamma')\(Gamma*X'))';
else
    % regularized
    epsS = 1e-6; % TODO
    S = ((Gamma*Gamma' + epsS*eye(K)) \ (Gamma*X'))';
end

end

