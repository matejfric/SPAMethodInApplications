% Never used, please see compute_L2()

function L = compute_L(C,Gamma,Lambda,X,alpha, PiY)
%COMPUTE_L Compute objective function value

[K,T] = size(Gamma);
[KY, KX] = size(Lambda); % KX = K

% 1st part of the objective function L (k-means)
L1 = 0;
for t=1:T
    L1_k = 0; % sum over k
    for k = 1:K
        L1_k = L1_k + (1/T)*Gamma(k,t) * ... % * norm ( X(:,t) , C(:,k) )
            dot(X(:,t) - C(:,k),X(:,t) - C(:,k)); % norm^2 = <*,*>
    end
    L1 = L1 + L1_k; % sum over t (outer sum)
end

% 2nd part of the objective function L (Lambda + Gamma)
L2 = 0;
for t=1:T
    L2_k = 0; % sum over k
    for k = 1:KY
        LambdaTimesGamma = Lambda(k,:) * Gamma(:,t);
        if PiY(k,t) ~= 0 % division by zero
            L2_k = L2_k - PiY(k,t) * log(LambdaTimesGamma / PiY(k,t));
        end
    end
    L2 = L2 + L2_k; % sum over t (outer sum)
end

L = alpha * L1 + (1 - alpha) * L2; % convex combination

end

