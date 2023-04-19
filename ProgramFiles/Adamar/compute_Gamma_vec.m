function [Gamma,it_in] = compute_Gamma_vec(C,Gamma,Lambda,X,epsilon, PiY)
%COMPUTE_GAMMA Adamar Gamma problem

[K,T] = size(Gamma);
D = size(X,1);

H = zeros(K,T);
for k = 1:K
    H(k,:) = sum(bsxfun(@minus, X, C(:,k)).^2,1);
end

spgoptions = spgOptions();
spgoptions.maxit = 5e2;
spgoptions.debug = false;
spgoptions.myeps = 1e-8;
spgoptions.alpha_min = 1e-6;
spgoptions.alpha_max = 1e6;

Gamma0 = Gamma;

f2 = @(Gamma) f_vec_spg(Gamma,Lambda,X,PiY,C,K,epsilon,T,D,H);
g2 = @(Gamma) g_vec_spg(Gamma,Lambda,X,PiY,C,K,epsilon,T,D,H);
p2 = @(Gamma) projection_simplex(Gamma);

f_old = sum(f2(Gamma0));
[Gamma,it_in] = spg_vec(@(Gamma) f2(Gamma),@(Gamma) g2(Gamma),@(Gamma) p2(Gamma),Gamma0,spgoptions);
f_new = sum(f2(Gamma));

if and(f_new > f_old, abs(f_new - f_old) > 1e-4)
        keyboard
end

end

function L = f_vec_spg(Gamma,Lambda,X,PiY,C,K,epsilon,Tcoeff,Dcoeff,H)
T = size(X,2);
KY = size(Lambda,1);
KX = size(Gamma,1);
L1 = zeros(1,T);
L2 = zeros(1,T);
logLambdaGamma = mylog(Lambda*Gamma);

L1 = (1/Tcoeff)*sum(Gamma.*H,1);

for ky = 1:KY
    PiYk = PiY(ky, :); 
    L2(PiYk ~= 0) = L2(PiYk ~= 0) - PiYk(PiYk ~= 0).*logLambdaGamma(ky,PiYk ~= 0);
end
L2 = (1/Dcoeff) * L2;

L = L1 + epsilon^2 * L2;

end

function G = g_vec_spg(Gamma,Lambda,X,PiY,C,K,epsilon,Tcoeff,Dcoeff,H)
T = size(X,2);
KY = size(Lambda,1);
G1 = zeros(K,T);
G2 = zeros(K,T);

LambdaGamma = Lambda*Gamma;
G1 = (1/Tcoeff)*H;
for kx = 1:K
    myval = zeros(1,T);
    for ky = 1:KY
        myval = myval + (PiY(ky,:)*Lambda(ky,kx))./max(LambdaGamma(ky,:),1e-12);
    end
    G2(kx,:) = -myval;
end
G2 = (1/Dcoeff) * G2;

G = G1 + epsilon^2 * G2;

end
