function [L,L1,L2] = compute_fval_adamar_kmeans(C,Gamma,Lambda,X,alpha, PiY, Tcoeff)
%COMPUTE_FVAL Compute objective function value (ADAMAR-KMEANS)

[KX,T] = size(Gamma);
[KY, KX] = size(Lambda); % KX = K

L = 0;
L1 = 0;
L2 = 0;
for t=1:T
    for kx = 1:KX
        L_lambda = 0;
        for ky = 1:KY
            L_lambda = L_lambda + PiY(ky,t) * log(max(Lambda(ky,kx),1e-12));
        end

        L1 = L1 + (1/Tcoeff) * Gamma(kx,t) * dot(X(:,t) - C(:,kx), X(:,t) - C(:,kx));
        L2 = L2 - L_lambda; 
    end
end

L = alpha*L1 + (1-alpha)*L2;

end

