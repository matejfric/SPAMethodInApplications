function Lambda = compute_Lambda_L2(Gamma,Lambda,PiY, Dcoeff)
%COMPUTE_LAMBDA_L2 Compute Lambda in L2 norm.

[KY,KX] = size(Lambda);

options = optimoptions(...
    'fmincon','Algorithm','interior-point',...
    'MaxFunctionEvaluations', 1.0e3, ...
    'ConstraintTolerance', 1e-4,...
    'Display', 'none'); % 'none', 'final', 'iter' (https://www.mathworks.com/help/optim/ug/fmincon.html#input_argument_options)

Aeq = kron(eye(KX),ones(1,KY));
beq = ones(KX,1);
lb = zeros(KX*KY,1);

lambda0 = reshape(Lambda,KY*KX,1);

f = @(lambda) f_fmincon(reshape(lambda,KY,KX),Gamma,PiY,Dcoeff);

tic
f_old = f(lambda0);
lambda = fmincon(@(lambda) f(lambda), lambda0, [],[], Aeq,beq, lb,[],[],options);
f_new = f(lambda);
time1 = toc;

Lambda = reshape(lambda,KY,KX);


if and(f_new > f_old, abs(f_new - f_old) > 1e-4)
    keyboard
end

end

function L2 = f_fmincon(Lambda,Gamma,PiY,Dcoeff)

L2 = (1/Dcoeff)*norm(PiY - Lambda*Gamma,'fro')^2;

end
