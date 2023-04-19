function [C, Gamma, PiX, Lambda, it, stats_train, L_out] = kmeans_lambda(X, PiY, K, alpha)
%KMEANS_KL Summary of this function goes here
arguments
    X {mustBeNumeric}
    PiY {double}
    K (1,1) {mustBeNumeric}
    alpha = 0.5
end

% K-means
[idx, C] = kmeans(X, K, 'MaxIter', 1000);
Gamma = zeros(K,length(idx));
for k = 1:K
   Gamma(k,idx==k) = 1; 
end

Lambda = lambda_solver_jensen(Gamma, PiY);

%PiX = round(Lambda*Gamma)'; % round => binary matrix
PiX = Lambda*Gamma;
stats_train = statistics(PiX(1,:), PiY(1,:));
% errX = sum(abs(PiX(:,1) - X(:,end)))/size(X,1);
% disp(['K-means+KL+Jensen learning error = ' num2str(errX)]);

it = NaN;
[L,L1,L2] = compute_L2(C',Gamma,Lambda,X',alpha,PiY,size(Gamma, 2),size(C',1));
L_out = struct('L', L, 'L1', L1, 'L2', L2);

end

