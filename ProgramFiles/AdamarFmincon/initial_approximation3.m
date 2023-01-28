function [Lambda, Gamma, C] = initial_approximation3(X, K, PiY)
%INITIAL_APPROXIMATION 

[rows,n] = size(X);
Gamma = rand(K, rows);
sumGamma = sum(Gamma,1);
for k = 1:K
    Gamma(k,:) = Gamma(k,:)./sumGamma;
end
C = randn(n-1,K); % randomly chosen initial centroids

Lambda = lambda_solver_jensen(Gamma, PiY);


end

