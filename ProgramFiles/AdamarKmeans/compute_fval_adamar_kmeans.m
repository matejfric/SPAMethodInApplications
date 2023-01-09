function L = compute_fval_adamar_kmeans(C,Gamma,Lambda,X,alpha, PiY)
%COMPUTE_FVAL Compute objective function value (ADAMAR-KMEANS)

[KX,T] = size(Gamma);
[KY, KX] = size(Lambda); % KX = K

L = 0;
for t=1:T
    for kx = 1:KX
        L_lambda = 0;
        for ky = 1:KY
            L_lambda = L_lambda + PiY(ky,t) * log(Lambda(ky,kx));
        end
        L = L + Gamma(kx,t) * (alpha * ... % * norm ( X(:,t) , C(:,k) )
            dot(X(:,t) - C(:,kx),X(:,t) - C(:,kx)) - ... % norm^2 = <*,*>
            (1-alpha) * L_lambda); 
    end
end

end

