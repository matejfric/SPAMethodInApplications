%--------------------------------------------------------------------------    
% Synthetic test
%--------------------------------------------------------------------------

clear all
addpath(genpath(pwd));
rng(13);
tic;

% Elapsed time was 4301.327156 seconds (~72 minutes). 

T = 1000; % This parameter significatly influences the runtime.
[X_true,Pi_true,C_true,Gamma_true,Lambda_true] = generate_binary_synthetic_problem(T);
%[X_true,Pi_true,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem(T);
M = size(Pi_true, 1); % Number of classification classes

% Add Gaussian noise
sigma = 0.15;
X = X_true + sigma * randn(size(X_true));

% Add artificial labelling errors
Pi = Pi_true;
nwrong = floor(0.05*T); 
rperm = randperm(T);
for i=1:nwrong
  idx1 = find(Pi(:,rperm(i))==1);
  idx2 = idx1;
  while idx1 == idx2
    idx2 = randi(M);
  end
  % Swap rows (labels)
  Pi([idx1 idx2],rperm(i)) = Pi([idx2 idx1],rperm(i)); 
end

maxIters = 50;
nrand = 3;
Ks = size(C_true,2);

alphas = 0:0.05:1;

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

        [C1, Gamma1, PiX1, Lambda1, it1, stats1, L_out1] = adamar_kmeans(X', Pi, K, alpha, maxIters, nrand);
        [C2, Gamma2, PiX2, Lambda2, it2, stats2, L_out2] = adamar_fmincon(X', Pi, K, alpha, maxIters, nrand);
        [C3, Gamma3, PiX3, Lambda3, it3, stats3, L_out3] = adamar_spa(X', Pi, K, alpha, maxIters, nrand);

        Ls{1}(idx_alpha,idx_K)  = L_out1.L;
        L1s{1}(idx_alpha,idx_K) = L_out1.L1;
        L2s{1}(idx_alpha,idx_K) = L_out1.L2;

        Ls{2}(idx_alpha,idx_K)  = L_out2.L;
        L1s{2}(idx_alpha,idx_K) = L_out2.L1;
        L2s{2}(idx_alpha,idx_K) = L_out2.L2;

        Ls{3}(idx_alpha,idx_K)  = L_out3.L;
        L1s{3}(idx_alpha,idx_K) = L_out3.L1;
        L2s{3}(idx_alpha,idx_K) = L_out3.L2;
    end
    
    clc % clear command window
    fprintf('Finished iteration for alpha=%.2f', alpha);
    pause(1)
end

%--------------------------------------------------------------------------    
%  Plots
%--------------------------------------------------------------------------

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

for idx_K=1:length(Ks)
    figure
    colororder(newcolors)
    ax = gca;
    ax.FontSize = axis_fs;
    
    subplot(2,2,1)
    title('L-curve', 'Interpreter','latex','FontSize', legend_fs)
    hold on
    plot(L1s{1}(:,idx_K), L2s{1}(:,idx_K),jensen_linestyle(:), 'LineWidth', lw);
    plot(L1s{2}(:,idx_K), L2s{2}(:,idx_K),spg_linestyle(:), 'LineWidth', lw);
    plot(L1s{3}(:,idx_K), L2s{3}(:,idx_K),spa_linestyle(:), 'LineWidth', lw);
    xlabel('$L_1^*$','Interpreter','latex','FontSize', label_fs)
    ylabel('$L_2^*$','Interpreter','latex','FontSize', label_fs)
    legend('$K$-means+KLD+Jensen','$K$-means+KLD','SPA+KLD','Interpreter','latex','FontSize', legend_fs)
    hold off
    grid on
    grid minor
    ax = gca;
    ax.FontSize = axis_fs;
    
    subplot(2,2,2)
    title('Objective Function Value', 'Interpreter','latex','FontSize', legend_fs)
    hold on
    grid on
    grid minor
    plot(alphas,Ls{1}(:, idx_K),jensen_linestyle(:), 'LineWidth', lw)
    plot(alphas,Ls{2}(:, idx_K),spg_linestyle(:), 'LineWidth', lw)
    plot(alphas,Ls{3}(:, idx_K),spa_linestyle(:), 'LineWidth', lw)
    xlabel('$\alpha$','Interpreter','latex','FontSize', label_fs)
    ylabel('$L$','Interpreter','latex','FontSize', label_fs)
    %legend('$K$-means+KLD+Jensen','$K$-means+KLD','SPA+KLD','Interpreter','latex','FontSize', legend_fs)
    
    subplot(2,2,3)
    title('$L_1$ Function Value', 'Interpreter','latex','FontSize', legend_fs)
    hold on
    grid on
    grid minor
    plot(alphas,L1s{1}(:, idx_K),jensen_linestyle(:), 'LineWidth', lw)
    plot(alphas,L1s{2}(:, idx_K),spg_linestyle(:), 'LineWidth', lw)
    plot(alphas,L1s{3}(:, idx_K),spa_linestyle(:), 'LineWidth', lw)
    xlabel('$\alpha$','Interpreter','latex','FontSize', label_fs)
    ylabel('$L_1$','Interpreter','latex','FontSize', label_fs)
    
    subplot(2,2,4)
    title('$L_2$ Function Value', 'Interpreter','latex','FontSize', legend_fs)
    hold on
    grid on
    grid minor
    plot(alphas,L2s{1}(:, idx_K),jensen_linestyle(:), 'LineWidth', lw)
    plot(alphas,L2s{2}(:, idx_K),spg_linestyle(:), 'LineWidth', lw)
    plot(alphas,L2s{3}(:, idx_K),spa_linestyle(:), 'LineWidth', lw)
    xlabel('$\alpha$','Interpreter','latex','FontSize', label_fs)
    ylabel('$L_2$','Interpreter','latex','FontSize', label_fs)
    hold off
end

toc

