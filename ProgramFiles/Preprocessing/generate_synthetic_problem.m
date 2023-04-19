function [X,Pi,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem(T)
%GENERATE_PROBLEM

K = 4; % clusters
C_true = randi([-10,10],10,K);
[D,K] = size(C_true);
  
Gamma_true = zeros(K,T);
idx = randi(K,T,1);
for k = 1:K
    Gamma_true(k,idx == k) = 1; % random Gamma
end

Lambda_true = [0.8 0   0   0.5
               0   1   0   0.4
               0   0   0.7 0 
               0.2 0   0.3 0.1];

Piprob = Lambda_true*Gamma_true;

edges = cumsum([zeros(1,T);Piprob],1);
uniform = rand(1, T);

idx = zeros(1,T);
for i = 1:size(Piprob,1)
    idx(uniform >= edges(i,:)) = idx(uniform >= edges(i,:)) + 1;
end

Pi = zeros(size(Piprob));
for i = 1:size(Pi,1)
    Pi(i,idx == i) = 1;
end

X = C_true*Gamma_true;
           
end

