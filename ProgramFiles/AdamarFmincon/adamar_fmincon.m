function [C, Gamma, PiX, Lambda, it, Lit, learningErrors] = ...
    adamar_fmincon(X, K, alpha, C0, Gamma0, Lambda0, PiY, trueLabels, maxIters)
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

disp('ADAMAR:')
if isempty(maxIters)
    maxIters = 1;
end
myeps = 1e-4;

% in this implementation I assume 1D data
[TRows, TCols] = size(X);
n = 1;

% set initial approximations
if isempty(C0)
    C = zeros(n,K);
else
    C = C0;
end

if isempty(Gamma0)
    Gamma = get_random(K, TRows);
else
    Gamma = Gamma0;
end

if isempty(Lambda0)
    Lambda = get_random(size(PiY, 1),K);
else
    Lambda = Lambda0;
end

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
    L = compute_L2(C,Gamma,Lambda,X,alpha, PiY);
    
    disp([' it=' num2str(it) ', L=' num2str(L)]);
    
    if abs(L - Lold) < myeps
        break; % stop outer "while" cycle
    end
    
    it = it + 1;
    
    Lit(it) = L; % for postprocessing
    
    % Computation of learning error
    PiX = round(Lambda*Gamma)'; % round => binary matrix
    learningErrors(it) = sum(abs(PiX(:,1) - trueLabels)) / length(trueLabels);
    disp(['Learning error = ' num2str(learningErrors(it))]);
end

% Never used:
% Random matrix <0,1> with columns that add up to 1 (non-uniform!)
function  Lambda = get_random(KY,KX)
Lambda = rand(KY,KX);
Lambda = bsxfun(@times, Lambda, 1./sum(Lambda,1)); % 1./sum(Lambda,1) is a row vector with length KX
% Equivalent: Lambda = Lambda * 1./sum(Lambda,1);
end

end

