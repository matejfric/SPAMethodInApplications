function [Lambda0, Gamma, C] = initial_approximation_plus_plus(X, K, PiY)
%INITIAL_APPROXIMATION 

C = kmeans_plus_plus(X',K);
%C = randn(size(X,2),K);

[rows,n] = size(X);

%Gamma = rand(K, rows);
%sumGamma = sum(Gamma,1);
%for k = 1:K
%    Gamma(k,:) = Gamma(k,:)./sumGamma;
%end

Gamma = zeros(K, rows);
idx = randi(K,rows,1);
for k = 1:K
    Gamma(k,idx == k) = 1; % random Gamma
end

Lambda0 = lambda_solver_jensen(Gamma, PiY);
% Lambda = compute_Lambda(Gamma,Lambda0,PiY);
% fprintf('||Lambda - Lambda0|| = %.2e\n', norm(Lambda - Lambda0, 'fro'))

end

