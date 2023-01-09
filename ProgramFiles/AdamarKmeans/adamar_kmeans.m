function [Lambda, C, K, stats]...
    = adamar_kmeans(X, K, alpha, maxIters)
%KMEANS_ADAMAR 
%   K........number of clusters

switch(nargin)
    case 2 
        alpha = 0.5;
        maxIters = 10;
    case 3
        maxIters = 10;
end

fprintf("\nPerforming the K-means algorithm for K=%d, alpha=%d\n", K, alpha);

PiY = [X(:,end)'; 1-X(:,end)']; % [P(x is corroded); P(x is not corroded)]
ground_truth = X(:,end);
X = X(:, 1:end-1);

% % Initial approximation of C and Lambda (Solution)
% [idx, C] = kmeans(X, K, 'MaxIter',1000);
% Gamma = zeros(K,length(idx));
% for k = 1:K
%     Gamma(k,idx==k) = 1;
% end
% Lambda = lambda_solver_jensen(Gamma, PiY);
% PiX = round(Lambda*Gamma)';

% INITIAL APPROXIMATIONS
[rows,~] = size(X);
sample_ids = zeros(K,1);
Gamma = zeros(K, rows);
for i = 1:K
    sample_ids(i) = randi(rows);
end
C = X(sample_ids,:); % randomly chosen initial centroids
for i = 1:rows
    Gamma(randi(K),i) = 1; % random Gamma
end
Lambda = lambda_solver_jensen(Gamma, PiY);

% Initial objective function value
L = compute_L2(C',Gamma,Lambda,X',alpha, PiY);
L0 = L;
fprintf("it=%d  L=%.2f\n", 0, L0);
learningErrors = zeros(0, maxIters); % preallocation
myeps = 1e-3; %TODO

T = size(X,1); % Number of features
for i = 1:maxIters
    %Compute Gamma
    Gamma = zeros(K, T);
    for t = 1:T
        PiYLambda = 0;
        for kx = 1:K
            for ky = 1:2
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
        ids = Gamma(k,:) == 1; % Matrix of indices of features affiliated to the k-th cluster
        C(k,:) = mean(X(ids,:));
    end
    
    % Is the objective function decreasing?
    L_old = L;
    L = compute_L2(C',Gamma,Lambda,X',alpha, PiY);
    if isnan(L) % Only one state is present in Lambda! bug?
        fprintf("\nObjective function value is NaN!\n")
        stats = statistics(zeros(numel(ground_truth),1), ground_truth);
        pause(0.1)
        break;
    end
    if abs(L - L_old) / (L0/10) < myeps
        break;
    end
    
    L_real = compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY);
    
    % Computation of learning error
    PiX = round(Lambda*Gamma)'; % Prediction (round => binary matrix)
    stats = statistics(PiX(:,1), ground_truth);
    learningErrors(i) = sum(abs(PiX(:,1) - ground_truth)) / length(ground_truth);
    fprintf("it=%d  L=%.2f  L_a=%.2f  FN=%d  FP=%d  f1score=%.3f  error:%.3f\n",...
        i, L_real, L, stats.fn, stats.fp, stats.f1score, learningErrors(i));
end
    
end

