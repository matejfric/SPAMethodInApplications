clear all
close all
addpath(genpath(pwd));

rng(42)

T = 1e3;
[X_true,Y_true,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem0(T);

sigma = 0.15; %0.15; % noise parameter
X = X_true + sigma * randn(size(X_true));

Y = Y_true;
% nwrong = floor(0.05*T); % wrong labels
% idx = randperm(T);
% for i=1:nwrong
%   Y(1:2,idx(i)) = Y(2:-1:1,idx(i));
% end

maxIters = 30;
nrand = 5;
Ks = size(C_true,2);

%alphas = 0:0.05:1;
alphas = 0:0.1:1;
%alphas = 0.05:0.05:0.95;

Ls = cell(3,1);
L1s = cell(3,1);
L2s = cell(3,1);
for i=1:3
    Ls{i}  = zeros(numel(alphas),length(Ks));
    L1s{i} = zeros(numel(alphas),length(Ks));
    L2s{i} = zeros(numel(alphas),length(Ks));
end

for idx_alpha=1:length(alphas)
    alpha = alphas(idx_alpha);
    
        for idx_K=1:length(Ks)
            K = Ks(idx_K);
    
            [C1, Gamma1, PiX1, Lambda1, it1, stats1, L_out1] = adamar_kmeans(X', Y, K, alpha, maxIters, nrand);
            [C2, Gamma2, PiX2, Lambda2, it2, stats2, L_out2] = adamar_fmincon(X', Y, K, alpha, maxIters, nrand);
            [C3, Gamma3, PiX3, Lambda3, it3, stats3, L_out3] = adamar_spa(X', Y, K, alpha, maxIters, nrand);

            Ls{1}(idx_alpha,idx_K)  = L_out1.L;
            L1s{1}(idx_alpha,idx_K) = L_out1.L1;
            L2s{1}(idx_alpha,idx_K) = L_out1.L2;

            Ls{2}(idx_alpha,idx_K)  = L_out2.L;
            L1s{2}(idx_alpha,idx_K) = L_out2.L1;
            L2s{2}(idx_alpha,idx_K) = L_out2.L2;

%             Ls{3}(idx_alpha,idx_K)  = L_out3.L;
%             L1s{3}(idx_alpha,idx_K) = L_out3.L1;
%             L2s{3}(idx_alpha,idx_K) = L_out3.L2;
            
        end
end

% L-curve
for idx_K=1:length(Ks)
    figure
    hold on
    title(sprintf('K=%d', K))
    plot(L1s{1}(:,idx_K), L2s{1}(:,idx_K),'r-o');
    plot(L1s{2}(:,idx_K), L2s{2}(:,idx_K),'b-o');
    plot(L1s{3}(:,idx_K), L2s{3}(:,idx_K),'m-o');
    xlabel('$L_1$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    legend('jensen','fmincon','spa')
    hold off
end

for idx_K=1:length(Ks)
    figure
    subplot(1,3,1)
    hold on
    plot(alphas,Ls{1}(:, idx_K),'r*-')
    plot(alphas,Ls{2}(:, idx_K),'b*-')
    plot(alphas,Ls{3}(:, idx_K),'m*-')
    xlabel('$\alpha$','Interpreter','latex')
    ylabel('$L$','Interpreter','latex')
    legend('jensen','fmincon','spa')
    subplot(1,3,2)
    hold on
    plot(alphas,L1s{1}(:, idx_K),'r*-')
    plot(alphas,L1s{2}(:, idx_K),'b*-')
    plot(alphas,L1s{3}(:, idx_K),'m*-')
    xlabel('$\alpha$','Interpreter','latex')
    ylabel('$L_1$','Interpreter','latex')
    subplot(1,3,3)
    hold on
    plot(alphas,L2s{1}(:, idx_K),'r*-')
    plot(alphas,L2s{2}(:, idx_K),'b*-')
    plot(alphas,L2s{3}(:, idx_K),'m*-')
    xlabel('$\alpha$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    hold off
end

function [X,Y,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem0(T)
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

%C_true = randi([-10,10],10,K);
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
