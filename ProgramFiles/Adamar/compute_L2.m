function [L,L1,L2] = compute_L2(C,Gamma,Lambda,X,alpha,PiY,Tcoeff,Dcoeff)
%COMPUTE_L2 Compute objective function value, vectorized version of compute_L  

[K,T] = size(Gamma);
[KY, KX] = size(Lambda); % KX = K

L1 = 0;
L2 = 0;
for k = 1:K
    L1 = L1 + (1/Tcoeff)*dot(Gamma(k,:),sum((X - kron(ones(1,T),C(:,k))).^2,1));
end

LambdaGamma = Lambda*Gamma;
for k = 1:KY
    PiYk = PiY(k, :); 
    L2 = L2 - dot(...
        PiYk(PiYk ~= 0),...
        mylog(LambdaGamma(k,PiYk ~= 0)./ PiYk(PiYk ~= 0)));
end
L2 = (1/Dcoeff) * L2;

L = alpha*L1 + (1-alpha)*L2;

end

