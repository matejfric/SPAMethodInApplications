function [Lambda0, Gamma, C] = initial_approximation_plus_plus(X, K, PiY)
%INITIAL_APPROXIMATION 

C = get_kmeans_pp_centroids(X',K); % fix: works correctly, but slowly
%C = kmeans_plus_plus(X',K); % may not be correct

[rows,n] = size(X);

%Gamma = rand(K, rows);
%sumGamma = sum(Gamma,1);
%for k = 1:K
%    Gamma(k,:) = Gamma(k,:)./sumGamma;
%end

Gamma = zeros(K, rows);
idx = randi(K,rows,1);
for k = 1:K
    Gamma(k,idx == k) = 1; % random binary Gamma
end

Lambda0 = lambda_solver_jensen(Gamma, PiY);
% Lambda = compute_Lambda(Gamma,Lambda0,PiY);
% fprintf('||Lambda - Lambda0|| = %.2e\n', norm(Lambda - Lambda0, 'fro'))

end

