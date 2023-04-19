function [mdl, L_out, stats_train] = train_klambda(X, PiY, K, epsilon)

%KMEANS_KL Summary of this function goes here
arguments
    X {mustBeNumeric}
    PiY {double}
    K (1,1) {mustBeNumeric}
    epsilon = 0.5
end

% K-means
[idx, C] = kmeans(X, K, 'MaxIter', 1000);
Gamma = zeros(K,length(idx));
for k = 1:K
   Gamma(k,idx==k) = 1; 
end

Lambda = lambda_solver_jensen(Gamma, PiY);

PiX = Lambda*Gamma;
stats_train = compute_training_stats(PiY, PiX);

it = NaN;
[L,L1,L2] = compute_L2(C',Gamma,Lambda,X',epsilon,PiY,size(Gamma, 2),size(C',1));
L_out = struct('L', L, 'L1', L1, 'L2', L2);

mdl = struct('C', C',...
    'Gamma', Gamma,...
    'Pi', PiX',...
    'Lambda', Lambda,...
    'it', it);

end

