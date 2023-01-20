function Gamma = compute_Gamma(C,Gamma,Lambda,X,alpha, PiY)
%COMPUTE_GAMMA Summary of this function goes here
%   Detailed explanation goes here
    [K,T] = size(Gamma);

    
    for t = 1:T
        gamma0 = Gamma(:,t);

        lb = zeros(size(gamma0));

        Aeq = ones(1,K);
        beq = 1;
    
        f = @(gamma) f_fmincon(gamma,Lambda,X,PiY,C,K,T,alpha); %compute_L2(C,reshape(gamma,T,K)',Lambda,X,alpha, PiY);
    
    options = optimoptions(...
        'fmincon','Algorithm','interior-point',...
        'Display','iter',...
        'MaxFunctionEvaluations', 2.0e4, ...
        'GradObj', 'on');
    
    tic
    gamma = fmincon(f,gamma0,[],[],Aeq,beq,lb,[],[],options);
    toc

    f_func = @(gamma) f_spg(gamma,Lambda,X,PiY,C,K,T,alpha);
    g_func = @(gamma) g_spg(gamma,Lambda,X,PiY,C,K,T,alpha);
    P_func = @(gamma) reshape(projection_simplex(reshape(gamma,T,K)')',K*T,1);
    
    options = spgOptions();
    options.debug = true;
    options.alpha_min = 1e-8;
    options.alpha_bb_init = 1e2;
    
    keyboard
    
    [gamma2,it,~] = spg(f_func,g_func,P_func,gamma0,options);   
    
    Gamma = reshape(gamma,T,K)';
end

function [L,g] = f_fmincon(gamma,Lambda,X,PiY,C,K,T,alpha)
    L = f_spg(gamma,Lambda,X,PiY,C,K,T,alpha);
    g = g_spg(gamma,Lambda,X,PiY,C,K,T,alpha);

end

function L = f_spg(gamma,Lambda,X,PiY,C,K,T,alpha)

Gamma = reshape(gamma,T,K)';
KY = size(Lambda,1);

L1 = 0;
L2 = 0;
for k = 1:K
    L1 = L1 + dot(Gamma(k,:),sum((X - kron(ones(1,T),C(:,k))).^2,1));
end

for k = 1:KY
    LambdaGamma = Lambda*Gamma;
    PiYk = PiY(k, :); 
    L2 = L2 - dot(...
        PiYk(PiYk ~= 0),...
        log(max(LambdaGamma(k,PiYk ~= 0),1e-12)));
end

L = alpha*L1 + (1-alpha)*L2;

end

function g = g_spg(gamma,Lambda,X,PiY,C,K,T,alpha)

Gamma = reshape(gamma,T,K)';
KY = size(Lambda,1);

G1 = zeros(K,T);
for kx = 1:K
   G1(kx,:) = sum((X - kron(ones(1,T),C(:,kx))).^2,1);
end

G2 = zeros(K,T);
for kx = 1:K
    for t = 1:T
        myval = 0;
        for m = 1:KY
            myval = myval + (PiY(m,t)*Lambda(m,kx))/max(Lambda(m,:)*Gamma(:,t),1e-12);
        end
        G2(kx,t) = myval;
    end
end

G = alpha*G1 + (1-alpha)*G2;

g = reshape(G',K*T,1);

end
