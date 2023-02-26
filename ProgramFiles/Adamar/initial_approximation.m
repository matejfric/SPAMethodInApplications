function [Lambda, Gamma, C] = initial_approximation(X, K, PiY)
%INITIAL_APPROXIMATION 

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

C=C';

end

