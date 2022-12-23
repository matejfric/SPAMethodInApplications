function Gamma = compute_Gamma(C,Gamma,Lambda,X,alpha, PiY)
%COMPUTE_GAMMA Summary of this function goes here
%   Detailed explanation goes here
    [K,T] = size(Gamma);
    gamma0 = reshape(Gamma',K*T,1);

    lb = zeros(size(gamma0));
    ub = ones(size(gamma0));

    % Equality constraints Aeq * \gamma = beq
    % i.e. sum of each column in Gamma equals 1
    % Aeq * gamma = ones()
    % gamma = vec(Gamma) %using reshape (x below represents one column)
%     [ 1      1     1     ]   [ x ]   [ 1 ]
%     |   1      1     1   | * | x | = | 1 ]
%     [     1      1     1 ]   [ x ]   [ 1 ]
    Aeq = kron(ones(1,K),eye(T));
    beq = ones(T,1);
    
    f = @(gamma) compute_L2(C,reshape(gamma,T,K)',Lambda,X,alpha, PiY);
    
    options = optimoptions(...
        'fmincon','Algorithm','interior-point',...
        'Display','iter',...
        'MaxFunctionEvaluations', 2.5e4);
    
    tic
    gamma = fmincon(f,gamma0,[],[],Aeq,beq,lb,ub,[],options);
    toc
    
    Gamma = reshape(gamma,T,K)';
end

