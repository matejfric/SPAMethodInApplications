function [C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats, L] = ...
    adamar_fmincon(X, K, alpha, maxIters)
%ADAMAR_FMINCON Summary of this function goes here
% X        data
% K        number of clusters
% alpha    penalty-regularisation parameter
% C        model parameters on each cluster (centroids)
% Gamma    probability indicator functions
% it       number of iterations
arguments
    X (:,:) double
    K {mustBeInteger}
    alpha double
    maxIters {mustBeInteger}
end

fprintf('ADAMAR, K=%d, alpha=%.2e\n', K, alpha);
if isempty(maxIters)
    maxIters = 1;
end

%SIMULATED ANNEALING
L.L = Inf;
Nrand = 5; % Number of random runs
for nrand = 1:Nrand
    disp(['- annealing run #' num2str(nrand)])
    PiY = [X(:,end), 1-X(:,end)]';
    [Lambda0, Gamma0, C0] = initial_approximation_plus_plus(X(:, 1:end-1), K, PiY);
    
    [C_temp, Gamma_temp, PiX_temp, Lambda_temp, it_temp, Lit_temp, learningErrors_temp, stats_temp, L_temp] =...
        adamar_fmincon_one(X, K, alpha, maxIters, Lambda0, Gamma0, C0);

    if L_temp.L < L.L
        C = C_temp;
        Gamma = Gamma_temp;
        PiX = PiX_temp;
        Lambda = Lambda_temp;
        it = it_temp;
        Lit = Lit_temp;
        learningErrors = learningErrors_temp;
        stats = stats_temp;
        L = L_temp;
    end
end

end

function [C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats, L] = ...
    adamar_fmincon_one(X, K, alpha, maxIters, Lambda0, Gamma0, C0)
%ADAMAR_FMINCON_ONE One run of adamar.

trueLabels = X(:,end);
PiY = [X(:,end), 1-X(:,end)]';
X = X(:, 1:end-1)';
T = size(X,2);

myeps = 1e-3;

Lambda = Lambda0;
Gamma = Gamma0;
C = C0;

% initial objective function value
L = realmax;

Lit = zeros(0, maxIters); % preallocation
learningErrors = zeros(0, maxIters); % preallocation
it = 0; % iteration counter

while it < maxIters % practical stopping criteria after computing new L (see "break")

    % compute Gamma
    disp(' - solving Gamma problem')
    Gamma = compute_Gamma(C,Gamma,Lambda,X,alpha, PiY);
    
    % compute C
    disp(' - solving C problem')
    C = compute_C(Gamma,X);

    % compute Lambda
    disp(' - solving Lambda problem')
    Lambda = compute_Lambda(Gamma,PiY);
    
    % compute objective function value
    Lold = L;
    [L, L1, L2] = compute_L2(C,Gamma,Lambda,X,alpha, PiY,T);
    
    disp([' it=' num2str(it) ', L=' num2str(L)]);
    
    if L > Lold
%        keyboard
    end
        
    if abs(L - Lold) < myeps
        break; % stop outer cycle
    end
    
    it = it + 1;
    
    Lit(it) = L; % for postprocessing
    
    % Computation of learning error
    % PiX = round(Lambda*Gamma)'; % round => binary matrix
    Gamma_rec = compute_Gamma_kmeans(C,X); % Reconstruction of Gamma
    PiX = round(Lambda*Gamma_rec)';
    learningErrors(it) = sum(abs(PiX(:,1) - trueLabels)) / length(trueLabels);
    stats(it) = statistics(PiX(:,1), trueLabels); %(labels, ground truth)
    fprintf('F1-Score: %.2f  |  Absolute error: %.2f\n', stats(it).f1score, learningErrors(it))

end

Ls.L = L;
Ls.L1 = L1;
Ls.L2 = L2;
L = Ls;

end

