function Lambda = compute_Lambda(Gamma,Lambda,PiY)
%COMPUTE_LAMBDA Summary of this function goes here
%    Lambda = lambda_solver_jensen(Gamma, PiY);

%Lambda0 = Lambda;

%[KY,KX] = size(Lambda);

%options = optimoptions(...
%    'fmincon','Algorithm','interior-point',...
%    'MaxFunctionEvaluations', 1.0e3, ... % Important hyperparameter!
%    'ConstraintTolerance', 1e-4,...
%    'Display', 'none'); % 'none', 'final', 'iter' (https://www.mathworks.com/help/optim/ug/fmincon.html#input_argument_options)
%        'GradObj', 'on',...
%        'HessianFcn', hessinterior);

%Aeq = kron(eye(KX),ones(1,KY));
%beq = ones(KX,1);
%lb = zeros(KX*KY,1);

%lambda0 = reshape(Lambda,KY*KX,1);
%tic
%f = @(lambda) f_fmincon(reshape(lambda,KY,KX),Gamma,PiY);
%time1 = toc;

%f_old = f(lambda0);
%lambda = fmincon(@(lambda) f(lambda), lambda0, [],[], Aeq,beq, lb,[],[],options);
%f_new = f(lambda);

%Lambda = reshape(lambda,KY,KX);

% SPG test
spgoptions = spgOptions();
spgoptions.maxit = 5e2;
spgoptions.debug = false;
spgoptions.myeps = 1e-8;
spgoptions.alpha_min = 1e-6;
spgoptions.alpha_max = 1e6;

f2 = @(Lambda) f_fmincon(Lambda,Gamma,PiY);
g2 = @(Lambda) g_spg(Lambda,Gamma,PiY);
p2 = @(Lambda) projection_simplex(Lambda);

%tic
f_old = f2(Lambda);
[Lambda,it_in] = spg(@(gamma) f2(gamma),@(gamma) g2(gamma),@(gamma) p2(gamma),Lambda,spgoptions);
f_new = f2(Lambda);
%time2 = toc;

if and(f_new > f_old, abs(f_new - f_old) > 1e-4)
    keyboard
end

end

function L2 = f_fmincon(Lambda,Gamma,PiY)

KY = size(PiY,1);

L2 = 0;
for k = 1:KY
    LambdaGamma = Lambda*Gamma;
    PiYk = PiY(k, :);
    L2 = L2 - dot(...
        PiYk(PiYk ~= 0),...
        mylog(LambdaGamma(k,PiYk ~= 0)));
end

end

function G = g_spg(Lambda,Gamma,PiY)

LambdaGamma = Lambda*Gamma;
G = -(PiY.*myinv(LambdaGamma))*Gamma';

end

function A = myinv(X)

A = 1./X;
A(isinf(A)) = 0; 

end
