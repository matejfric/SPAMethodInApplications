function [L,L1,L2] = compute_fval_adamar_kmeans(C,Gamma,Lambda,X,alpha, PiY, Tcoeff)
%COMPUTE_FVAL Compute objective function value (ADAMAR-KMEANS)

[KX,T] = size(Gamma);

L1 = 0;
for k = 1:KX
    L1 = L1 + (1/Tcoeff)*dot(Gamma(k,:),sum((X - kron(ones(1,T),C(:,k))).^2,1));
end

L2 = -sum(sum((PiY*Gamma').*mylog(Lambda)));

L = alpha*L1 + (1-alpha)*L2;

end

