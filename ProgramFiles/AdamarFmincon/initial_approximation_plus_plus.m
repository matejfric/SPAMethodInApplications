function [Lambda, Gamma, C] = initial_approximation_plus_plus(X, K, PiY)
%INITIAL_APPROXIMATION 

C = kmeans_plus_plus(X',K);

[rows,n] = size(X);
Gamma = rand(K, rows);
sumGamma = sum(Gamma,1);
for k = 1:K
    Gamma(k,:) = Gamma(k,:)./sumGamma;
end

Lambda = lambda_solver_jensen(Gamma, PiY);


end

