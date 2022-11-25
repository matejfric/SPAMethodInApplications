function [] = adamar_helper(X, K)
%ADAMAR_HELPER Summary of this function goes here
%   K........number of clusters

% Remember scaling for testing dataset
a = min(X(:, 1));
b = max(X(:, 1));
X(:, 1) = (X(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

% Initial approximation of C and Lambda
[idx, C] = kmeans(X(:,1:4), K, 'MaxIter',1000);
Gamma = zeros(K,length(idx));
for k = 1:K
    Gamma(k,idx==k) = 1;
end
PiY = [X(:,5)'; 1-X(:,5)'];
Lambda = lambda_solver_jensen(Gamma, PiY);

% PiY (2 x T), resp. (K_Y, T)
% Gamma=PiX (K, T), resp. (K_X, T)
% Lambda (2 x K), resp. (K_Y, K_X)
% C (K, 4), resp. (K, features)
maxIters = 1;
[C, Gamma, PiX, Lambda, it, Lit, learningErrors] = ...
    adamar_fmincon(X(:,1:4)', 10, 0.5, C', Gamma, Lambda, PiY, X(:,5), maxIters);

Lambda

testingErrors = zeros(0,10);
fprintf("Testing errors: ")
for i= 1:size(testingErrors,2)
    testingErrors(i) = adamar_predict(Lambda, C, K, a, b, i);
    fprintf("%.2f, ", testingErrors(i));
end
fprintf("testing precision: %.2f", sum(testingErrors) / size(testingErrors,2));

end

