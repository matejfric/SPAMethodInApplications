function [Lambda, C, K, a, b, stats]...
    = adamar_kmeans_v2(X, K, alpha, maxIters)
%KMEANS_ADAMAR Summary of this function goes here
%   Detailed explanation goes here
%   K........number of clusters

switch(nargin)
    case 2 
        alpha = 0.5;
        maxIters = 10;
    case 3
        maxIters = 10;
end

fprintf("\nPerforming the K-means algorithm for K=%d\n", K);

PiY = [X(:,end)'; 1-X(:,end)']; % [P(x is corroded); P(x is not corroded)]
ground_truth = X(:,end);
X = X(:, 1:end-1);

% Remember scaling for testing dataset
a = min(X(:, 1));
b = max(X(:, 1));
% X(:, 1) = (X(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

% Initial approximation of C and Lambda
[idx, C] = kmeans(X, K, 'MaxIter',1000);
Gamma = zeros(K,length(idx));
for k = 1:K
    Gamma(k,idx==k) = 1;
end
Lambda = lambda_solver_jensen(Gamma, PiY);
PiX = round(Lambda*Gamma)';

% Initial objective function value
%L = realmax;
L = compute_L2(C',Gamma,Lambda,X',alpha, PiY);
Lit = zeros(0, maxIters); % preallocation
learningErrors = zeros(0, maxIters); % preallocation
myeps = 1e-6;

T = size(X,1); % Number of features
i=1;
for i = 1:maxIters % redundant value of Lambda doesn't change!
    %Compute Gamma
    Gamma = zeros(K, T);
    for t = 1:T
        PiYLambda = 0;
        for kx = 1:K
            for ky = 1:2
                %PiYLambda = PiYLambda + PiY(ky,t)*(Lambda(ky,kx) - 1); % log ~~ x-1
                PiYLambda = PiYLambda + PiY(ky,t)*log(Lambda(ky,kx));
            end
        end
        expression = alpha * (C - X(t,:) ).^2 - (1-alpha)*PiYLambda;
        [~,id] = min( sum( expression , 2 )); % min(sum of rows)
        Gamma(id,t) = 1;
    end
    
    % Update Lambda
    Lambda = lambda_solver_jensen(Gamma, PiY);
    
    % Update C
    for k = 1:K
        ids = Gamma(k,:) == 1; % Matrix of indices of features affiliated k-th cluster
        C(k,:) = mean(X(ids,:));
    end
    
    % Is the objective function decreasing?
    % Compute function value
    L_old = L;
    L = compute_L2(C',Gamma,Lambda,X',alpha, PiY);
    disp([' it=' num2str(i) ', L=' num2str(L)]);
%     if abs(L - L_old) < myeps
%          break; % stop outer "while" cycle
%     end
    % Computation of learning error
    PiX = round(Lambda*Gamma)'; % round => binary matrix
    learningErrors(i) = sum(abs(PiX(:,1) - ground_truth)) / length(ground_truth);
    disp(['Learning error = ' num2str(learningErrors(i))]);
end

stats = statistics(PiX(:,1), ground_truth);
    
end

