function [Lambda, C, K, a, b] = adamar_kmeans(X, K, alpha, maxIters)
%KMEANS_ADAMAR Summary of this function goes here
%   Detailed explanation goes here
%   K........number of clusters

switch(nargin)
    case 2 
        alpha = 0.5;
        maxIters = 1000;
    case 3
        maxIters = 1000;
end

fprintf("\nPerforming the K-means algorithm for K=%d\n", K);

PiY = [X(:,end)'; 1-X(:,end)']; % [P(x is corroded); P(x is not corroded)]
X = X(:, 1:end-1); % Drop ground truth

% Remember scaling for testing dataset
a = min(X(:, 1));
b = max(X(:, 1));
X(:, 1) = (X(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

% Initial approximation of C and Lambda
[idx, C] = kmeans(X, K, 'MaxIter',1000); % X without ground truth
Gamma = zeros(K,length(idx));
for k = 1:K
    Gamma(k,idx==k) = 1;
end
Lambda = lambda_solver_jensen(Gamma, PiY);

T = size(X,1); % Number of features
for i = progress(1:maxIters)
    Gamma = zeros(K, T);
    for t = 1:T
%       expression1 = alpha * (C - X(t,:) ).^2 - ((1-alpha)*PiY(:,t)'*(Lambda - 1))'; % log ~~ x-1
        PiYLambda = 0;
        for kx = 1:K
            for ky = 1:2
                %PiYLambda = PiYLambda + PiY(ky,t)*(Lambda(ky,kx) - 1);
                PiYLambda = PiYLambda + PiY(ky,t)*log(Lambda(ky,kx));
            end
        end
        expression = alpha * (C - X(t,:) ).^2 - (1-alpha)*PiYLambda;
%         if expression1 ~= expression
%             keyboard
%         end
        [~,id] = min( sum( expression , 2 )); % min(sum of rows)
        Gamma(id,t) = 1;
    end
    
    % Update Lambda
    Lambda = lambda_solver_jensen(Gamma, PiY);
    
    for k = 1:K
        ids = (Gamma(k,:) == 1); % Matrix of indices of features affiliated k-th cluster
        C(k,:) = mean(X(ids,:));
    end
end
    
end

