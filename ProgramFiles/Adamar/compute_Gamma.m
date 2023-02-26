function Gamma = compute_Gamma(C,Gamma,Lambda,X,alpha, PiY)
%COMPUTE_GAMMA Adamar Gamma problem

[K,T] = size(Gamma);

spgoptions = spgOptions();
spgoptions.maxit = 5e2;
spgoptions.debug = false;
spgoptions.myeps = 1e-8;
spgoptions.alpha_min = 1e-6;
spgoptions.alpha_max = 1e6;

for t = progress(1:T)
    
    gamma0 = Gamma(:,t);

    f2 = @(gamma) f_spg(gamma,Lambda,X(:,t),PiY(:,t),C,K,alpha,T);
    g2 = @(gamma) g_spg(gamma,Lambda,X(:,t),PiY(:,t),C,K,alpha,T);
    p2 = @(gamma) projection_simplex(gamma);
    
    f_old = f2(gamma0);
    [Gamma(:,t),it_in] = spg(@(gamma) f2(gamma),@(gamma) g2(gamma),@(gamma) p2(gamma),gamma0,spgoptions);
    f_new = f2(Gamma(:,t));

    if and(f_new > f_old, abs(f_new - f_old) > 1e-4)
        keyboard
    end

end

end

function [L,g] = f_fmincon(gamma,Lambda,x,piY,C,K,alpha,Tcoeff)
%F_FMINCON Objective function

KY = size(Lambda,1);

L1 = 0;
L2 = 0;
LambdaGamma = Lambda*gamma;
G1 = zeros(K,1);
G2 = zeros(K,1);

for kx = 1:K
    L1 = L1 + (1/Tcoeff)*gamma(kx)*sum((x - C(:,kx)).^2);
    G1(kx,:) = (1/Tcoeff)*sum((x - C(:,kx)).^2,1);
    
    myval = 0;
    for ky = 1:KY
        myval = myval + (piY(ky)*Lambda(ky,kx))/max(Lambda(ky,:)*gamma,1e-12);
    end
    G2(kx) = -myval;
end

for ky = 1:KY
    PiYk = piY(ky);
    if PiYk ~= 0
        L2 = L2 - PiYk*mylog(LambdaGamma(ky));
    end
end

L = alpha*L1 + (1-alpha)*L2;
g = alpha*G1 + (1-alpha)*G2;

end

function L = f_spg(gamma,Lambda,x,piY,C,K,alpha,Tcoeff)
%F_SPG Objective function

KY = size(Lambda,1);

L1 = 0;
L2 = 0;
LambdaGamma = Lambda*gamma;

for kx = 1:K
    L1 = L1 + (1/Tcoeff)*gamma(kx)*sum((x - C(:,kx)).^2);
end

for ky = 1:KY
    PiYk = piY(ky);
    if PiYk ~= 0
        L2 = L2 - PiYk*mylog(LambdaGamma(ky));
    end
end

L = alpha*L1 + (1-alpha)*L2;

end

function g = g_spg(gamma,Lambda,x,piY,C,K,alpha,Tcoeff)
%G_SPG Gradient function

KY = size(Lambda,1);

G1 = zeros(K,1);
G2 = zeros(K,1);

for kx = 1:K
    G1(kx,:) = (1/Tcoeff)*sum((x - C(:,kx)).^2,1);
    
    myval = 0;
    for ky = 1:KY
        myval = myval + (piY(ky)*Lambda(ky,kx))/max(Lambda(ky,:)*gamma,1e-12);
    end
    G2(kx) = -myval;
end

g = alpha*G1 + (1-alpha)*G2;

end

function H = hessinterior_spg(gamma,Lambda, x, piY, C, K, alpha, lambda)
%HESSINTERIOR_SPG Hessian

KY = size(Lambda,1);

H = zeros(K,K);
for k_hat = 1:K
    for k_tilde = 1:K
        for m = 1:KY
            H(k_hat,k_tilde) = H(k_hat,k_tilde) + ...
                (piY(m)*Lambda(m,k_hat)*Lambda(m,k_tilde))/max((Lambda(m,:)*gamma)^2,1e-12);
        end
    end
end
H = (1-alpha)*H;

end

