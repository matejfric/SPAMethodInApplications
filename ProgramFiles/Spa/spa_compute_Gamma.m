function [Gamma,it_in] = spa_compute_Gamma(C,Gamma,Lambda,X,epsilon, PiY)
%COMPUTE_GAMMA Adamar Gamma problem

[K,T] = size(Gamma);
D = size(X,1);

spgoptions = spgOptions();
spgoptions.maxit = 5e2;
spgoptions.debug = false;
spgoptions.myeps = 1e-8;
spgoptions.alpha_min = 1e-6;
spgoptions.alpha_max = 1e6;

it_in = 0;
for t = 1:T
    
    gamma0 = Gamma(:,t);

    f2 = @(gamma) f_spg(gamma,Lambda,X(:,t),PiY(:,t),C,K,epsilon,T,D);
    g2 = @(gamma) g_spg(gamma,Lambda,X(:,t),PiY(:,t),C,K,epsilon,T,D);
    p2 = @(gamma) projection_simplex(gamma);
    
    f_old = f2(gamma0);
    [Gamma(:,t),it_in2] = spg(@(gamma) f2(gamma),@(gamma) g2(gamma),@(gamma) p2(gamma),gamma0,spgoptions);
    f_new = f2(Gamma(:,t));

    if and(f_new > f_old, abs(f_new - f_old) > 1e-4)
        %keyboard
        break;
    end

    it_in = max(it_in2,it_in);
end

end

function L = f_spg(gamma,Lambda,x,piY,S,K,epsilon,Tcoeff,Dcoeff)
%F_SPG Objective function

KY = size(Lambda,1);

L1 = (1/Tcoeff)*norm(x - S*gamma,2)^2;
L2 = 0;
LambdaGamma = Lambda*gamma;

for ky = 1:KY
    PiYk = piY(ky);
    if PiYk ~= 0
        L2 = L2 - PiYk*mylog(LambdaGamma(ky));
    end
end
L2 = (1/Dcoeff) * L2;

L = L1 + epsilon^2 * L2;

end

function g = g_spg(gamma,Lambda,x,piY,S,K,epsilon,Tcoeff,Dcoeff)
%G_SPG Gradient function

KY = size(Lambda,1);

G1 = (1/Tcoeff)*(2*(S'*S)*gamma - 2*S'*x);
G2 = zeros(K,1);

for kx = 1:K
    
    myval = 0;
    for ky = 1:KY
        myval = myval + (piY(ky)*Lambda(ky,kx))/max(Lambda(ky,:)*gamma,1e-12);
    end
    G2(kx) = -myval;
end
G2 = (1/Dcoeff) * G2;

g = G1 + epsilon^2 * G2;

end
