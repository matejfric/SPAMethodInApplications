function Gamma = compute_Gamma(C,Gamma,Lambda,X,myeps, PiY)
%COMPUTE_GAMMA Adamar Gamma problem

[K,T] = size(Gamma);

lb = zeros(K,1); % Upper bound is redundant, see equality constraints

Aeq = ones(1,K);
beq = 1;

for t = progress(1:T)
    % Hessian for fmincon()
    hessinterior = @(gamma,lambda) hessinterior_spg(gamma,Lambda,X(:,t),PiY(:,t),C,K,myeps,lambda);

    options = optimoptions(...
        'fmincon','Algorithm','interior-point',...
        'MaxFunctionEvaluations', 1.0e3, ... % Important hyperparameter!
        'ConstraintTolerance', 1e-4,...
        'Display', 'none',... % 'none', 'final', 'iter' (https://www.mathworks.com/help/optim/ug/fmincon.html#input_argument_options)
        'GradObj', 'on',...
        'HessianFcn', hessinterior);
    
    gamma0 = Gamma(:,t);
    f = @(gamma) f_fmincon(gamma,Lambda,X(:,t),PiY(:,t),C,K,myeps,T);
    
    fold = f(Gamma(:,t));
    Gamma(:,t) = fmincon(f,gamma0,[],[],Aeq,beq,lb,[],[],options);
    fnew = f(Gamma(:,t));

    if fnew > fold
        %TODO: hotfix
        Gamma(:,t) = gamma0;
    end
end

end

function [L,g] = f_fmincon(gamma,Lambda,x,piY,C,K,myeps,Tcoeff)
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
        L2 = L2 - PiYk*log(max(LambdaGamma(ky),1e-12));
    end
end

L = L1 + myeps*L2;
g = G1 + myeps*G2;

end

function H = hessinterior_spg(gamma,Lambda, x, piY, C, K, myeps, lambda)
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
H = myeps*H;

end

