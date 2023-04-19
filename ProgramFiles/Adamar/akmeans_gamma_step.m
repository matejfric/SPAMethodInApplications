function [Gamma] = akmeans_gamma_step(X, C, K, Lambda, PiY, epsilon, Tcoeff, Dcoeff)
%AKMEANS_GAMMA_STEP Compute Gamma

[T,~] = size(X);
Lambda_hat = mylog(Lambda);
expressions = zeros(T,K);

for kx = 1:K
    L1Gamma_kx = (1/Tcoeff) * sum((X - kron(ones(T,1),C(kx,:))).^2,2);
    L2Gamma_kx = (1/Dcoeff) * -Lambda_hat(:,kx)'*PiY;

    expressions(:,kx) = L1Gamma_kx + epsilon^2*L2Gamma_kx';
end

[~,id] = min(expressions,[],2);
Gamma = zeros(K, T);

for kx = 1:K
   Gamma(kx,id == kx) = 1; 
end

end

