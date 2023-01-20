function [C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats, L] = ...
    adamar_fmincon(X, K, alpha, maxIters)

%ADAMAR_FMINCON Summary of this function goes here
% X        data
% K        number of clusters
% alpha    penalty-regularisation parameter
% C        model parameters on each cluster (centroids)
% Gamma    probability indicator functions
% it       number of iterations

% arguments
%     X (:,:) double
%     K double
%     alpha double
%     C0 (:,:) double = zeros(size(X,1),K)
%     Gamma0 (:,:) double = get_random(K,size(X,2))
%     Lambda0 (:,:) double = get_random(K,K)
% end

fprintf('ADAMAR, K=%d, alpha=%e\n', K, alpha);
if isempty(maxIters)
    maxIters = 1;
end
myeps = 1e-4;

% set initial approximations
trueLabels = X(:,end);
PiY = [X(:,end), 1-X(:,end)]';
[Lambda, Gamma, C] = initial_approximation(X, K, PiY);
X = X(:, 1:end-1)';

% initial objective function value
L = realmax;

Lit = zeros(0, maxIters); % preallocation
learningErrors = zeros(0, maxIters); % preallocation
it = 0; % iteration counter

while it < maxIters % practical stopping criteria after computing new L (see "break")
    
    % compute C
    %if isempty(C0)
    disp(' - solving C problem')
    C = compute_C(Gamma,X);
    %end
    
    % compute Lambda
    %if isempty(Lambda0)
    disp(' - solving Lambda problem')
    Lambda = compute_Lambda(Gamma,PiY);
    %end
    
    % compute Gamma
    %if isempty(Gamma0)
    disp(' - solving Gamma problem')
    Gamma = compute_Gamma(C,Gamma,Lambda,X,alpha, PiY);
    %end
    
    % compute function value
    Lold = L;
    [L, L1, L2] = compute_L2(C,Gamma,Lambda,X,alpha, PiY);
    
    disp([' it=' num2str(it) ', L=' num2str(L)]);
    
    if abs(L - Lold) < myeps
        break; % stop outer "while" cycle
    end
    
    it = it + 1;
    
    Lit(it) = L; % for postprocessing
    
    % Computation of learning error
    PiX = round(Lambda*Gamma)'; % round => binary matrix
    learningErrors(it) = sum(abs(PiX(:,1) - trueLabels)) / length(trueLabels);
    stats(it) = statistics(PiX(:,1), trueLabels); %(labels, ground_truth)
    fprintf('F1-Score: %.2f  |  Absolute error: %.2f\n', stats(it).f1score, learningErrors(it))
end

Ls.L = L;
Ls.L1 = L1;
Ls.L2 = L2;
L = Ls;

end

