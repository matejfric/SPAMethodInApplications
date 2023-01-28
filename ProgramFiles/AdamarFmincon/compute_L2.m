function [L,L1,L2] = compute_L2(C,Gamma,Lambda,X,alpha,PiY,Tcoeff)
%COMPUTE_L2 Compute objective function value, vectorized version of compute_L  

[K,T] = size(Gamma);
[KY, KX] = size(Lambda); % KX = K

L1 = 0;
L2 = 0;
for k = 1:K
    L1 = L1 + (1/Tcoeff)*dot(Gamma(k,:),sum((X - kron(ones(1,T),C(:,k))).^2,1));
end
for k = 1:KY
    LambdaGamma = Lambda*Gamma;
    PiYk = PiY(k, :); 
    L2 = L2 - dot(...
        PiYk(PiYk ~= 0),...
        log(max(LambdaGamma(k,PiYk ~= 0),1e-12)./ PiYk(PiYk ~= 0)));
end

L = alpha*L1 + (1-alpha)*L2;

end

