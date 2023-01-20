function [Lambda, Gamma, C] = initial_approximation2(X, K, PiY)
%INITIAL_APPROXIMATION2 

[idx, C] = kmeans(X(:,1:end-1), K, 'MaxIter',1000);
Gamma = zeros(K,length(idx));
for k = 1:K
    Gamma(k,idx==k) = 1;
end
Lambda = lambda_solver_jensen(Gamma, PiY);

C=C';

end

