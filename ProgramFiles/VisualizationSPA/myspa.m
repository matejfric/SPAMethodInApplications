function [S, Gamma, it, L] = myspa(X, K, maxIters, Nrand)
%SPA Scalable probabilistic approximation
%   X...DxT matrix
arguments
    X, K, maxIters = 100, Nrand = 3
end

%SIMULATED ANNEALING
L = Inf;
for nrand = 1:Nrand
    disp(['- annealing run #' num2str(nrand)])
    
    [Gamma0, S0] = myspa_initial_approximation(X, K);
    
    [S_temp, Gamma_temp, it_temp, L_temp]=spa_one(X, Gamma0, S0, maxIters);
    
    if L_temp < L
        S = S_temp;
        Gamma = Gamma_temp;
        it = it_temp;
        L = L_temp;
    end
end

end

function [S, Gamma, it, L]=spa_one(X, Gamma0, S0, maxIters)

bugfix = true;

myeps = 1e-3;

Gamma = Gamma0;
S = S0;

[~, Tcoeff] = size(X);
Tcoeff = 1;

% initial objective function value
L = realmax;

it = 0; % iteration counter

while it < maxIters % practical stopping criteria after computing new L (see "break")
    
    L_old = L;
    
    if ~bugfix; disp(' - solving Gamma problem'); end
    if bugfix; fprintf(' - before Gamma:    %.2f\n', myspa_compute_L(S,Gamma,X,Tcoeff)); end
    
    %Gamma = myspa_compute_Gamma(S,Gamma,X);
    %[Gamma, ~] = myspa_compute_Gamma_spg(S,Gamma,X);
    [Gamma, ~] = myspa_compute_Gamma_vec(S,Gamma,X);
    
    if bugfix; fprintf(' - after Gamma:     %.2f\n', myspa_compute_L(S,Gamma,X,Tcoeff)); end
        
    S = myspa_compute_S(Gamma,X);
    
    if bugfix; fprintf(' - after S:         %.2f\n', myspa_compute_L(S,Gamma,X,Tcoeff)); end

    L = myspa_compute_L(S,Gamma,X,Tcoeff);
    
    if abs(L - L_old) < myeps
        break; 
    end
    
    if L > L_old
       if bugfix
            %keyboard
            break;
       end
    end
    
    it = it + 1;
    
end

end

