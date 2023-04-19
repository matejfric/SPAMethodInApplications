clear all
%close all
addpath(genpath(pwd));
rng(13);
tic;

T = 1000;
[X_true,Pi_true,C_true,Gamma_true,Lambda_true] = generate_binary_synthetic_problem(T);
%[X_true,Pi_true,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem(T);
M = size(Pi_true, 1); % number of classification classes

sigma = 0.15; % noise parameter
X = X_true + sigma * randn(size(X_true));

Pi = Pi_true;
nwrong = floor(0.05*T); % wrong labels
rperm = randperm(T);

for i=1:nwrong
  idx1 = find(Pi(:,rperm(i))==1);
  idx2 = idx1;
  while idx1 == idx2
    idx2 = randi(M);
  end
  Pi([idx1 idx2],rperm(i)) = Pi([idx2 idx1],rperm(i)); % swap rows (labels)
end

maxIters = 50;
nrand = 10;
Ks = size(C_true,2);

epsilons = 10.^(-6:1:6);

Ls = cell(3,1);
L1s = cell(3,1);
L2s = cell(3,1);
for i=1:3
    Ls{i}  = zeros(numel(epsilons),length(Ks));
    L1s{i} = zeros(numel(epsilons),length(Ks));
    L2s{i} = zeros(numel(epsilons),length(Ks));
end

for idx_epsilon=1:length(epsilons)
    epsilon = epsilons(idx_epsilon);
    
    for idx_K=1:length(Ks)
        K = Ks(idx_K);

        [C1, Gamma1, PiX1, Lambda1, it1, stats1, L_out1] = adamar_kmeans(X', Pi, K, epsilon, maxIters, nrand);
        [C2, Gamma2, PiX2, Lambda2, it2, stats2, L_out2] = adamar_fmincon(X', Pi, K, epsilon, maxIters, nrand);
%         [C3, Gamma3, PiX3, Lambda3, it3, stats3, L_out3] = adamar_spa(X', Y, K, epsilon, maxIters, nrand);

        Ls{1}(idx_epsilon,idx_K)  = L_out1.L;
        L1s{1}(idx_epsilon,idx_K) = L_out1.L1;
        L2s{1}(idx_epsilon,idx_K) = L_out1.L2;

        Ls{2}(idx_epsilon,idx_K)  = L_out2.L;
        L1s{2}(idx_epsilon,idx_K) = L_out2.L1;
        L2s{2}(idx_epsilon,idx_K) = L_out2.L2;

%         Ls{3}(idx_epsilon,idx_K)  = L_out3.L;
%         L1s{3}(idx_epsilon,idx_K) = L_out3.L1;
%         L2s{3}(idx_epsilon,idx_K) = L_out3.L2;
    end
    
%     clc % clear command window
    fprintf('Finished iteration for epsilon=%.2f', epsilon);
%     pause(1)
end

% PLOTS
lw = 2; % LineWidth
label_fs = 15;
axis_fs = 12;
legend_fs = 14;
jensen_linestyle = '*-';
spg_linestyle = 'd-';
spa_linestyle = 's-';
newcolors = [0     0.447 0.741 %blue
             0.85  0.325 0.098 %orange
             0.466 0.674 0.188 %green
             ];
% L-curve
for idx_K=1:length(Ks)
    figure
    colororder(newcolors)
    hold on
    %title(sprintf('K=%d', K))
    plot(L1s{1}(:,idx_K), L2s{1}(:,idx_K),jensen_linestyle(:), 'LineWidth', lw);
    plot(L1s{2}(:,idx_K), L2s{2}(:,idx_K),spg_linestyle(:), 'LineWidth', lw);
    plot(L1s{3}(:,idx_K), L2s{3}(:,idx_K),spa_linestyle(:), 'LineWidth', lw);
    xlabel('$L_1$','Interpreter','latex','FontSize', label_fs)
    ylabel('$L_2$','Interpreter','latex','FontSize', label_fs)
    legend('jensen','spg','spa','FontSize', legend_fs)
    set(gca, 'XScale', 'log')
    set(gca, 'YScale', 'log')
    hold off
    grid on
    grid minor
    ax = gca;
    ax.FontSize = axis_fs;
end

for idx_K=1:length(Ks)
    figure
    colororder(newcolors)
    ax = gca;
    ax.FontSize = axis_fs;
    
    subplot(1,3,1)
    hold on
    grid on
    grid minor
    plot(epsilons,Ls{1}(:, idx_K),jensen_linestyle(:), 'LineWidth', lw)
    plot(epsilons,Ls{2}(:, idx_K),spg_linestyle(:), 'LineWidth', lw)
    plot(epsilons,Ls{3}(:, idx_K),spa_linestyle(:), 'LineWidth', lw)
    xlabel('$\epsilon$','Interpreter','latex','FontSize', label_fs)
    ylabel('$L$','Interpreter','latex','FontSize', label_fs)
    set(gca, 'XScale', 'log')
    legend('jensen','spg','spa','FontSize', legend_fs)
    
    subplot(1,3,2)
    hold on
    grid on
    grid minor
    plot(epsilons,L1s{1}(:, idx_K),jensen_linestyle(:), 'LineWidth', lw)
    plot(epsilons,L1s{2}(:, idx_K),spg_linestyle(:), 'LineWidth', lw)
    plot(epsilons,L1s{3}(:, idx_K),spa_linestyle(:), 'LineWidth', lw)
    xlabel('$\epsilon$','Interpreter','latex','FontSize', label_fs)
    ylabel('$L_1$','Interpreter','latex','FontSize', label_fs)
    set(gca, 'XScale', 'log')
    
    subplot(1,3,3)
    hold on
    grid on
    grid minor
    plot(epsilons,L2s{1}(:, idx_K),jensen_linestyle(:), 'LineWidth', lw)
    plot(epsilons,L2s{2}(:, idx_K),spg_linestyle(:), 'LineWidth', lw)
    plot(epsilons,L2s{3}(:, idx_K),spa_linestyle(:), 'LineWidth', lw)
    xlabel('$\epsilon$','Interpreter','latex','FontSize', label_fs)
    ylabel('$L_2$','Interpreter','latex','FontSize', label_fs)
    set(gca, 'XScale', 'log')
    hold off
end

toc

