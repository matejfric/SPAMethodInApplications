function [C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats, L] = ...
    adamar_fmincon(X, PiY, K, alpha, maxIters, Nrand)
%ADAMAR_FMINCON Summary of this function goes here
% X        data
% K        number of clusters
% alpha    penalty-regularisation parameter
% C        model parameters on each cluster (centroids)
% Gamma    probability indicator functions
% it       number of iterations
arguments
    X (:,:) double
    PiY (:,:) double
    K {mustBeInteger}
    alpha double
    maxIters {mustBeInteger} = 100;
    Nrand {mustBeInteger} = 5; % Number of random runs
end

fprintf('ADAMAR, K=%d, alpha=%.2e\n', K, alpha);
if isempty(maxIters)
    maxIters = 1;
end

%SIMULATED ANNEALING
L.L = Inf;
for nrand = 1:Nrand
    disp(['- annealing run #' num2str(nrand)])
    %    PiY = [X(:,end), 1-X(:,end)]';
    [Lambda0, Gamma0, C0] = initial_approximation_plus_plus(X, K, PiY);
    
    [C_temp, Gamma_temp, PiX_temp, Lambda_temp, it_temp, Lit_temp, learningErrors_temp, stats_temp, L_temp] =...
        adamar_fmincon_one(X, PiY, K, alpha, maxIters, Lambda0, Gamma0, C0);
    
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
    adamar_fmincon_one(X, PiY, K, alpha, maxIters, Lambda0, Gamma0, C0)
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

Lit = zeros(0, maxIters); % preallocation
learningErrors = zeros(0, maxIters); % preallocation
it = 0; % iteration counter

while it < maxIters % practical stopping criteria after computing new L (see "break")
    
    % GAMMA
    if ~bugfix; disp(' - solving Gamma problem'); end
    if bugfix; fprintf(' - before Gamma:    %.2f\n', compute_L2(C,Gamma,Lambda,X,alpha, PiY,T,D)); end
    
    Gamma0 = Gamma;
    %tic
    [Gamma,~] = compute_Gamma_vec(C,Gamma0,Lambda,X,alpha,PiY);
    %time1 = toc
    
%     tic
%     [Gamma2,it2] = compute_Gamma(C,Gamma0,Lambda,X,alpha,PiY);
%     time2 = toc;
    
    if bugfix; fprintf(' - after Gamma:     %.2f\n', compute_L2(C,Gamma,Lambda,X,alpha, PiY,T,D)); end
    
    % C
    if ~bugfix; disp(' - solving C problem'); end
    if bugfix; fprintf(' - before C:        %.2f\n', compute_L2(C,Gamma,Lambda,X,alpha, PiY,T,D)); end
    
    C = compute_C(Gamma,X);
    
    if bugfix; fprintf(' - after C:         %.2f\n', compute_L2(C,Gamma,Lambda,X,alpha, PiY,T,D)); end
    
    % LAMBDA
    if ~bugfix; disp(' - solving Lambda problem'); end
    if bugfix; fprintf(' - before Lambda:   %.2f\n', compute_L2(C,Gamma,Lambda,X,alpha, PiY,T,D)); end
    
    Lambda = compute_Lambda(Gamma,Lambda,PiY,D);
    %Lambda = lambda_solver_jensen(Gamma, PiY);
    
    if bugfix; fprintf(' - after Lambda:    %.2f\n', compute_L2(C,Gamma,Lambda,X,alpha, PiY,T,D)); end

    % Compute objective function value
    Lold = L;
    [L, L1, L2] = compute_L2(C,Gamma,Lambda,X,alpha,PiY,T,D);
    
    disp([' it=' num2str(it) ', L=' num2str(L)]);
    
    if abs(L - Lold) < myeps
        break; % stop outer cycle
    end
    
    if L > Lold
       if bugfix
            keyboard
       end
    end
    
    it = it + 1;
    
    Lit(it) = L; % for postprocessing
    
    % PiX = round(Lambda*Gamma)'; % round => binary matrix
    Gamma_rec = compute_Gamma_kmeans(C,X); % Reconstruction of Gamma
    PiX = round(Lambda*Gamma_rec)';
    [stats(it)] = compute_stats(PiY, PiX);
    fprintf('F1-Score: %.2f\n', stats(it).f1score)
    
end

Ls.L = L;
Ls.L1 = L1;
Ls.L2 = L2;
L = Ls;

end

function [stats] = compute_stats(PiY, PiX)

if size(PiY, 1) > 2 % multi-class classification
    c = size(PiY, 1); % number of classes
    PiX = round(PiX');
    R = PiX(:, sum(PiX,1)==0 | sum(PiX,1) > 1);
    r = randi([1 c],1,size(R,2));
    PiX(:, sum(PiX,1)==0 | sum(PiX,1) > 1) = bsxfun(@eq, r(:), 1:c)';
    [prediction, ~] = find(PiX);
    [ground_truth, ~] = find(round(PiY));
    if length(prediction) ~= length(ground_truth)
        keyboard
    end
    stats = statistics_multiclass(prediction, ground_truth);
    
else % binary classification
    ground_truth = PiY(1,:)';
    stats = statistics(PiX(:,1), ground_truth);
%   learningErrors(i) = sum(abs(PiX(:,1) - ground_truth')) / length(ground_truth);
end

end
