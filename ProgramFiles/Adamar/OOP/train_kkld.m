function [mdl,L,stats] = train_kkld(...
    X, PiY, K, epsilon, maxIters, Nrand, scaleT, verbose)
%TRAIN_KKLD 
% X        data
% K        number of clusters
% epsilon    penalty-regularisation parameter
% C        model parameters on each cluster (centroids)
% Gamma    probability indicator functions
% it       number of iterations
arguments
    X (:,:) double
    PiY (:,:) double
    K {mustBeInteger}
    epsilon double
    maxIters {mustBeInteger} = 100;
    Nrand {mustBeInteger} = 5; % Number of random runs
    scaleT = true;
    verbose = false;
end
if verbose
fprintf('ADAMAR, K=%d, epsilon=%.2e\n', K, epsilon);
end
if isempty(maxIters)
    maxIters = 1;
end

%SIMULATED ANNEALING
L.L = Inf;
for nrand = 1:Nrand
    if verbose
        disp(['- annealing run #' num2str(nrand)])
    end

    [Lambda0, Gamma0, C0] = initial_approximation_plus_plus(X, K, PiY);
    
    [C_temp, Gamma_temp, PiX_temp, Lambda_temp, it_temp, stats_temp, L_temp] =...
        adamar_fmincon_one(X, PiY, K, epsilon, maxIters, Lambda0, Gamma0, C0, verbose);
    
    if L_temp.L < L.L
        C = C_temp;
        Gamma = Gamma_temp;
        PiX = PiX_temp;
        Lambda = Lambda_temp;
        it = it_temp;
        stats = stats_temp;
        L = L_temp;
    end
end

mdl = struct('C', C,...
    'Gamma', Gamma,...
    'Pi', PiX',...
    'Lambda', Lambda,...
    'it', it);

end

function [C, Gamma, PiX, Lambda, it, stats, L] = ...
    adamar_fmincon_one(X, PiY, K, epsilon, maxIters, Lambda0, Gamma0, C0, verbose)
%ADAMAR_FMINCON_ONE One run of adamar.

bugfix = false;

[T, D] = size(X);
X = X';

myeps = 1e-3;

Lambda = Lambda0;
Gamma = Gamma0;
C = C0;

% initial objective function value
L = realmax;

it = 0; % iteration counter

while it < maxIters % practical stopping criteria after computing new L (see "break")
    
    % GAMMA
    if verbose
        if ~bugfix
            disp(' - solving Gamma problem'); 
        end
        if bugfix
            fprintf(' - before Gamma:    %.2f\n',...
                compute_L2(C,Gamma,Lambda,X,epsilon, PiY,T,D));
        end
    end
    Gamma0 = Gamma;
    [Gamma,~] = compute_Gamma_vec(C,Gamma0,Lambda,X,epsilon,PiY);
    if bugfix; fprintf(' - after Gamma:     %.2f\n', compute_L2(C,Gamma,Lambda,X,epsilon, PiY,T,D)); end
    
    % CENTROIDS
    if verbose
        if ~bugfix
            disp(' - solving C problem');
        end
        if bugfix
            fprintf(' - before C:        %.2f\n',...
                compute_L2(C,Gamma,Lambda,X,epsilon, PiY,T,D)); 
        end
    end
    C = compute_C(Gamma,X);
    if bugfix; fprintf(' - after C:         %.2f\n', compute_L2(C,Gamma,Lambda,X,epsilon, PiY,T,D)); end
    
    % LAMBDA
    if verbose
        if ~bugfix
            disp(' - solving Lambda problem'); 
        end
        if bugfix 
            fprintf(' - before Lambda:   %.2f\n',...
                compute_L2(C,Gamma,Lambda,X,epsilon, PiY,T,D)); 
        end
    end
    Lambda = compute_Lambda(Gamma,Lambda,PiY,D);
    if bugfix; fprintf(' - after Lambda:    %.2f\n', compute_L2(C,Gamma,Lambda,X,epsilon, PiY,T,D)); end

    % Compute objective function value
    Lold = L;
    [L, L1, L2] = compute_L2(C,Gamma,Lambda,X,epsilon,PiY,T,D);
    
    if verbose
        disp([' it=' num2str(it) ', L=' num2str(L)]);
    end
    
    if abs(L - Lold) < myeps
        break; % stop outer cycle
    end
    
    if L > Lold
       keyboard
    end
    
    it = it + 1;
    
    % PiX = round(Lambda*Gamma)'; % round => binary matrix
    Gamma_rec = compute_Gamma_kmeans(C,X); % Reconstruction of Gamma
    PiX = Lambda*Gamma_rec;
    [stats] = compute_training_stats(PiY, PiX);
    if verbose
        fprintf('F1-Score: %.2f\n', stats.f1score)
    end
end

Ls.L = L;
Ls.L1 = L1;
Ls.L2 = L2;
L = Ls;

end


