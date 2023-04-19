function [L,L1,L2] = compute_fval_adamar_kmeans(C,Gamma,Lambda,X,epsilon, PiY, Tcoeff, Dcoeff)
%COMPUTE_FVAL Compute objective function value (ADAMAR-KMEANS)

[KX,T] = size(Gamma);

L1 = 0;
for k = 1:KX
    L1 = L1 + (1/Tcoeff)*dot(Gamma(k,:),sum((X - kron(ones(1,T),C(:,k))).^2,1));
end

L2 = (1/Dcoeff) * -sum(sum((PiY*Gamma').*mylog(Lambda)));

L = L1 + epsilon^2 * L2;

end

