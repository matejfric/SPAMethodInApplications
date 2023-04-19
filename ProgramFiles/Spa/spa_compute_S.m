function S = spa_compute_S(Gamma,X)
%SPA_COMPUTE_S Calculation of the SPA cluster parameters
n = size(X,1);
K = size(Gamma,1);

if false
    S = ((Gamma*Gamma')\(Gamma*X'))';
else
    % regularized
    epsS = 1e-6; % TODO
    S = ((Gamma*Gamma' + epsS*eye(K))\(Gamma*X'))';
end

end

