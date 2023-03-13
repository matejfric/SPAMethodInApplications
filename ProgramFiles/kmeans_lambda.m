function [C, Gamma, PiX, Lambda, it, stats_train, L_out] = kmeans_lambda(X, K, PiY)
%KMEANS_KL Summary of this function goes here
arguments
    X {mustBeNumeric}
    K (1,1) {mustBeNumeric}
    PiY {double}
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
L_out = NaN;

end

