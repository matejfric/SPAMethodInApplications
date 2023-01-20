function Gamma = compute_Gamma(C,Gamma,Lambda,X,alpha, PiY)
%COMPUTE_GAMMA Summary of this function goes here
%   Detailed explanation goes here
[K,T] = size(Gamma);

lb = zeros(K,1);

Aeq = ones(1,K);
beq = 1;

tic
for t = 1:T
    hessinterior = @(gamma,lambda) hessinterior_spg(gamma,Lambda,X(:,t),PiY(:,t),C,K,alpha,lambda);

    options = optimoptions(...
        'fmincon','Algorithm','interior-point',...
        'Display','iter',...
        'MaxFunctionEvaluations', 2.0e4, ...
        'Display', 'none',...
        'GradObj', 'on');
%        'HessianFcn',hessinterior);
    
    gamma0 = Gamma(:,t);
    f = @(gamma) f_fmincon(gamma,Lambda,X(:,t),PiY(:,t),C,K,alpha);
    Gamma(:,t) = fmincon(f,gamma0,[],[],Aeq,beq,lb,[],[],options);
end
toc

end

function [L,g] = f_fmincon(gamma,Lambda,x,piY,C,K,alpha)

KY = size(Lambda,1);

L1 = 0;
L2 = 0;
LambdaGamma = Lambda*gamma;
G1 = zeros(K,1);
G2 = zeros(K,1);

for kx = 1:K
    L1 = L1 + gamma(kx)*sum((x - C(:,kx)).^2);
    G1(kx,:) = sum((x - C(:,kx)).^2,1);

    myval = 0;
    for ky = 1:KY
        myval = myval + (piY(ky)*Lambda(ky,kx))/max(Lambda(ky,:)*gamma,1e-12);
    end
    G2(kx) = -myval;
end

for ky = 1:KY
    PiYk = piY(ky);
    if PiYk ~= 0
        L2 = L2 - PiYk*log(max(LambdaGamma(ky),1e-12));
    end
end

L = alpha*L1 + (1-alpha)*L2;
g = alpha*G1 + (1-alpha)*G2;

end

function L = f_spg(gamma,Lambda,x,piY,C,K,alpha)

KY = size(Lambda,1);

L1 = 0;
L2 = 0;
for k = 1:K
    L1 = L1 + gamma(k)*sum((x - C(:,k)).^2);
end

LambdaGamma = Lambda*gamma;
for k = 1:KY
    PiYk = piY(k);
    if PiYk ~= 0
        L2 = L2 - PiYk*log(max(LambdaGamma(k),1e-12));
    end
end

L = alpha*L1 + (1-alpha)*L2;

end

function g = g_spg(gamma,Lambda,x,piY,C,K,alpha)

KY = size(Lambda,1);

G1 = zeros(K,1);
for kx = 1:K
    G1(kx,:) = sum((x - C(:,kx)).^2,1);
end

G2 = zeros(K,1);
for kx = 1:K
    myval = 0;
    for m = 1:KY
        myval = myval + (piY(m)*Lambda(m,kx))/max(Lambda(m,:)*gamma,1e-12);
    end
    G2(kx) = -myval;
end

G = alpha*G1 + (1-alpha)*G2;

g = G;%reshape(G',K*T,1);

end

function H = hessinterior_spg(gamma,Lambda,x,piY,C,K,alpha,lambda)

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

