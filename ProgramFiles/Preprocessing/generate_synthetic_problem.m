function [X,Y,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem(T)
%GENERATE_PROBLEM

rng(42);

C_true = ...
    [ 1 0 1 0; ...
      0 2 3 -2; ...
      1 1 -2 0; ...
      0 0 5 1; ...
      0 -1 2 1];
  
% C_true = ...
% [ 10 7 13 25; ...
%   117 -42 39 -2; ...
%   11 8 -27 0; ...
%   90 2 5 36; ...
%   9 -10 27 77];

K = 4; % clusters
c = 3; % classes

C_true = randi([-10,10],10,K);
% r = randi([1 c],1,K);
% Lambda_true = bsxfun(@eq, r(:), 1:c)';
% C_true = normalize(C_true);

[D,K] = size(C_true);
  
Gamma_true = zeros(K,T);
idx = randi(K,T,1);
for k = 1:K
    Gamma_true(k,idx == k) = 1; % random Gamma
end

Lambda_true = [1 0 0 1;...
               0 1 0 0; ...
               0 0 1 0];

Y = Lambda_true*Gamma_true; % PiY

X = C_true*Gamma_true;
           
end

