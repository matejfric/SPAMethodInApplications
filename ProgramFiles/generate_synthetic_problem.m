function [X,Y,C_true,Gamma_true,Lambda_true] = generate_problem(T)
%GENERATE_PROBLEM Summary of this function goes here
%   Detailed explanation goes here

C_true = ...
    [ 1 0 1 0; ...
      0 2 3 -2; ...
      1 1 -2 0; ...
      0 0 5 1; ...
      0 -1 2 1];

[D,K] = size(C_true);
  
Gamma_true = zeros(K,T);
idx = randi(K,T,1);
for k = 1:K
    Gamma_true(k,idx == k) = 1; % random Gamma
end

Lambda_true = [1 0 0 0.5;...
               0 1 0 0.5; ...
               0 0 1 0];

Y = Lambda_true*Gamma_true;

X = C_true*Gamma_true;
           
end

