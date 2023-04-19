function [L,L1,L2] = spa_compute_L(S,Gamma,Lambda,X,epsilon,PiY,Tcoeff,Dcoeff)
%SPA_COMPUTE_L Compute objective function value for SPA

[K,T] = size(Gamma);
[KY, KX] = size(Lambda); % KX = K

L1 = (1/Tcoeff)*norm(X - S*Gamma,'fro')^2;
L2 = 0;

LambdaGamma = Lambda*Gamma;
for k = 1:KY
    PiYk = PiY(k, :); 
    L2 = L2 - dot(...
        PiYk(PiYk ~= 0),...
        mylog(LambdaGamma(k,PiYk ~= 0)./ PiYk(PiYk ~= 0)));
end
L2 = (1/Dcoeff) * L2;

L = L1 + epsilon^2 * L2;

end

