function [Lambda, C, Gamma, K, stats, L_out]...
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

idx = randi(K,rows,1);
for i = 1:K
    Gamma(i,idx == i) = 1; % random Gamma
end
Lambda = lambda_solver_jensen(Gamma, PiY);

% Initial objective function value
L = compute_L2(C',Gamma,Lambda,X',alpha, PiY);
L0 = L;
fprintf("it=%d  L=%.2f\n", 0, L0);
learningErrors = zeros(0, maxIters); % preallocation
myeps = 1e-6; %TODO

T = size(X,1); % Number of features
for i = 1:maxIters
    
    %disp([' - before Gamma: ' num2str(compute_L2(C',Gamma,Lambda,X',alpha, PiY))])
    
    %Compute Gamma
    Lambda_hat = log(max(Lambda,1e-12));
    expressions = zeros(T,K);
    for kx = 1:K
        L1Gamma_kx = sum((X - kron(ones(T,1),C(kx,:))).^2,2);
        L2Gamma_kx = -Lambda_hat(:,kx)'*PiY;
        
        expressions(:,kx) = alpha*L1Gamma_kx + (1-alpha)*L2Gamma_kx';
    end
    [~,id] = min(expressions,[],2);
    Gamma = zeros(K, T);
    for kx = 1:K
       Gamma(kx,id == kx) = 1; 
    end
    
    %disp([' - after Gamma: ' num2str(compute_L2(C',Gamma,Lambda,X',alpha, PiY))])
    
    % Update Lambda
    Lambda = lambda_solver_jensen(Gamma, PiY);

    %disp([' - before C: ' num2str(compute_L2(C',Gamma,Lambda,X',alpha, PiY))])
    
    % Update C
    for k = 1:K
        ids = Gamma(k,:) == 1; % Matrix of indices of features affiliated to the k-th cluster
        if sum(ids) > 0 
            C(k,:) = mean(X(ids,:));
        end
    end

    %disp([' - L = ' num2str(compute_L2(C',Gamma,Lambda,X',alpha, PiY)) ', L_real: ' num2str(compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY))])
    
    % Is the objective function decreasing?
    L_old = L;
    [L,L1,L2] = compute_L2(C',Gamma,Lambda,X',alpha, PiY);
    if isnan(L) % Only one state is present in Lambda! bug?
        fprintf("\nObjective function value is NaN!\n")
        
        keyboard
        
        stats = statistics(zeros(numel(ground_truth),1), ground_truth);
        pause(0.1)
        break;
    end
%    if abs(L - L_old) / (L0/10) < myeps
    if abs(L - L_old) < myeps
        break;
    end
    
    [L_real] = compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY);
    
%    if isnan(L_real)
%        keyboard
%    end
        
    % Computation of learning error
    PiX = round(Lambda*Gamma)'; % Prediction (round => binary matrix)
    stats = statistics(PiX(:,1), ground_truth);
    learningErrors(i) = sum(abs(PiX(:,1) - ground_truth)) / length(ground_truth);
    fprintf("it=%d  L=%.2f  L_a=%.2f  FN=%d  FP=%d  f1score=%.3f  error:%.3f\n",...
        i, L_real, L, stats.fn, stats.fp, stats.f1score, learningErrors(i));
end

L_out.L = L_real;
L_out.L1 = L1;
L_out.L2 = L2;

end

