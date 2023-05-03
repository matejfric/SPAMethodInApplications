function Gamma = myspa_compute_Gamma(S,Gamma,X)
%SPA_COMPUTE_GAMMA_FMINCON

[K,T] = size(Gamma);

options = optimoptions(...
    'fmincon','Algorithm','interior-point',...
    'MaxFunctionEvaluations', 1.0e4, ...
    'ConstraintTolerance', 1e-4,...
    'Display', 'final'); % 'none', 'final', 'iter' (https://www.mathworks.com/help/optim/ug/fmincon.html#input_argument_options)

% Equality constraints Aeq * \gamma = beq
% i.e. sum of each column in Gamma equals 1
% Aeq * gamma = ones()
% gamma = vec(Gamma) %using reshape (x below represents one column)
%     [ 1      1     1     ]   [ x ]   [ 1 ]
%     |   1      1     1   | * | x | = | 1 |
%     [     1      1     1 ]   [ x ]   [ 1 ]

Aeq = kron(ones(1,K),eye(T));
beq = ones(T,1);
lb = zeros(T*K,1);
ub = ones(T*K,1);

gamma0 = reshape(Gamma,K*T,1);

L = @(gamma) L_fmincon(reshape(gamma,K,T),X,S,T);

L_old = L(gamma0);
gamma = fmincon(@(gamma) L(gamma), gamma0, [],[], Aeq,beq, lb,ub,[],options);
L_new = L(gamma);

Gamma = reshape(gamma,K,T);

if and(L_new > L_old, abs(L_new - L_old) > 1e-4)
    %keyboard
    return
end

end

function L = L_fmincon(Gamma, X, S, T)

L = (1/T) * norm(X - S*Gamma,'fro')^2;

end
