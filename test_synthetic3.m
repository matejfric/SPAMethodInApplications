clear all
%close all
addpath(genpath(pwd));

% PLOTS
lw = 2; % LineWidth
label_fs = 15;
axis_fs = 12;
legend_fs = 14;
kmeans_lambda_linestyle = 's-';
jensen_linestyle = '*-';
spg_linestyle = 'd-';
spa_linestyle = 's-';
newcolors = [0     0.447 0.741 %blue
    0.85  0.325 0.098 %orange
    0.466 0.674 0.188 %green
    ];

rng(13); %For reproducibility

T = ceil(1e3*(5/3));
[X_true,Y_true,C_true,Gamma_true,Lambda_true] = generate_binary_synthetic_problem(T);
%[X_true,Y_true,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem(T);

M = size(Y_true,1);
sigma = 0; %0.15; % noise parameter
X = X_true + sigma * randn(size(X_true));

Y = Y_true;
nwrong = floor(0.0*T); % wrong labels
rperm = randperm(T);
for i=1:nwrong
    idx1 = find(Y(:,rperm(i))==1);
    idx2 = idx1;
    while idx1 == idx2
        idx2 = randi(M);
    end
    Y([idx1 idx2],rperm(i)) = Y([idx2 idx1],rperm(i)); % swap rows (labels)
end

Ttrain = floor(0.6*T);
Ttest = floor(0.2*T);
Tval = floor(0.2*T);

Xdata = X(:,1:Ttrain+Ttest);
Ydata = Y(:,1:Ttrain+Ttest);

Xval = X(:,(Ttrain+Ttest+1):end);
Yval = Y(:,(Ttrain+Ttest+1):end);

maxIters = 50;
nrand = 4;
Ks = 4;%[2,3,4,5,6]; %size(C_true,2);

%alphas = 0:0.005:0.02;
%alphas = 0:0.2:1;
%alphas = 0.05:0.05:0.95;
%alphas = 0.9:0.01:0.99;
%alphas = 0:0.05:1;

epss = 10.^[-7:1:7];
alphas = 1./(epss + 1);

Nrand = 1;

errstrain_rand = cell(3,1);
scoretrain_rand = cell(3,1);
errstest_rand = cell(3,1);
scoretest_rand = cell(3,1);
for i=1:3
    errstrain_rand{i}  = Inf*ones(numel(alphas),length(Ks),Nrand);
    scoretrain_rand{i} = 0*ones(numel(alphas),length(Ks),Nrand);
    errstest_rand{i}  = Inf*ones(numel(alphas),length(Ks),Nrand);
    scoretest_rand{i} = 0*ones(numel(alphas),length(Ks),Nrand);
end


for myrand = 1:Nrand
    idx = randperm(Ttrain+Ttest);
    Xperm = Xdata(:,idx);
    Yperm = Ydata(:,idx);
    
    Xtrain = Xperm(:,1:Ttrain);
    Xtest = Xperm(:,(Ttrain+1):end);
    Ytrain = Yperm(:,1:Ttrain);
    Ytest = Yperm(:,(Ttrain+1):end);
    
    % here store L values for various alpha and K
    Ls = cell(3,1);
    L1s = cell(3,1);
    L2s = cell(3,1);
    for i=1:3
        Ls{i}  = zeros(numel(alphas),length(Ks));
        L1s{i} = zeros(numel(alphas),length(Ks));
        L2s{i} = zeros(numel(alphas),length(Ks));
    end
    
    
    for idx_K=1:length(Ks)
        K = Ks(idx_K);
        % solve problem for various alpha
        for idx_alpha=1:length(alphas)
            alpha = alphas(idx_alpha);

            [C1, Gamma1, PiX1, Lambda1, it1, stats1, L_out1] = ...
                kmeans_lambda(Xtrain', Ytrain, K, alpha);
            C1 = C1';
            
            [C2, Gamma2, PiX2, Lambda2, it2, stats2, L_out2] = ...
                adamar_kmeans(Xtrain', Ytrain, K, alpha, maxIters, nrand);
            C2 = C2';
            
            [C3, Gamma3, PiX3, Lambda3, it3, stats3, L_out3] = ...
                adamar_fmincon(Xtrain', Ytrain, K, alpha, maxIters, nrand);
            PiX3 = PiX3';

            if norm(PiX2 - Lambda2*Gamma2) > 1e-4
                keyboard
            end
            
            %           [C3, Gamma3, PiX3, Lambda3, it3, stats3, L_out3] = adamar_spa(X', Y, K, alpha, maxIters, nrand);
            
            Ls{1}(idx_alpha,idx_K)  = L_out1.L;
            L1s{1}(idx_alpha,idx_K) = L_out1.L1;
            L2s{1}(idx_alpha,idx_K) = L_out1.L2;
            errstrain_rand{1}(idx_alpha,idx_K,myrand) = norm(Lambda1*Gamma1 - Ytrain,'fro');
            scoretrain_rand{1}(idx_alpha,idx_K,myrand) = stats1.f1score;
            
            Ls{2}(idx_alpha,idx_K)  = L_out2.L;
            L1s{2}(idx_alpha,idx_K) = L_out2.L1;
            L2s{2}(idx_alpha,idx_K) = L_out2.L2;
            errstrain_rand{2}(idx_alpha,idx_K,myrand) = norm(Lambda2*Gamma2 - Ytrain,'fro');
            scoretrain_rand{2}(idx_alpha,idx_K,myrand) = stats2.f1score;

            Ls{3}(idx_alpha,idx_K)  = L_out3.L;
            L1s{3}(idx_alpha,idx_K) = L_out3.L1;
            L2s{3}(idx_alpha,idx_K) = L_out3.L2;
            errstrain_rand{3}(idx_alpha,idx_K,myrand) = norm(Lambda3*Gamma3 - Ytrain,'fro');
            scoretrain_rand{3}(idx_alpha,idx_K,myrand) = stats3.f1score;
            
            % apply model to validation data
            Gammatest1 = compute_Gamma_kmeans(C1,Xtest);
            Ytest_est1 = Lambda1*Gammatest1;
            errstest_rand{1}(idx_alpha,idx_K,myrand) = norm(Ytest_est1 - Ytest,'fro');
            [stats1] = statistics_multiclass(myonehotdecode(Ytest_est1',1:M,1), myonehotdecode(Ytest',1:M,1));
            scoretest_rand{1}(idx_alpha,idx_K,myrand) = stats1.f1score;
            
            Gammatest2 = compute_Gamma_kmeans(C2,Xtest);
            Ytest_est2 = Lambda2*Gammatest2;
            errstest_rand{2}(idx_alpha,idx_K,myrand) = norm(Ytest_est2 - Ytest,'fro');
            [stats2] = statistics_multiclass(myonehotdecode(Ytest_est2',1:M,1), myonehotdecode(Ytest',1:M,1));
            scoretest_rand{2}(idx_alpha,idx_K,myrand) = stats2.f1score;

            Gammatest3 = compute_Gamma_kmeans(C3,Xtest);
            Ytest_est3 = Lambda3*Gammatest3;
            errstest_rand{3}(idx_alpha,idx_K,myrand) = norm(Ytest_est3 - Ytest,'fro');
            [stats3] = statistics_multiclass(myonehotdecode(Ytest_est3',1:M,1), myonehotdecode(Ytest',1:M,1));
            scoretest_rand{3}(idx_alpha,idx_K,myrand) = stats3.f1score;
            

        end % alpha
        
    end % K
    
    if myrand == 1
        % L-curve
        for idx_K=1:length(Ks)
            figure
            colororder(newcolors)
            hold on
            title(sprintf('K=%d', Ks(idx_K)))
            plot(L1s{1}(:,idx_K), L2s{1}(:,idx_K),kmeans_lambda_linestyle(:), 'LineWidth', lw);
            plot(L1s{2}(:,idx_K), L2s{2}(:,idx_K),jensen_linestyle(:), 'LineWidth', lw);
            plot(L1s{3}(:,idx_K), L2s{3}(:,idx_K),spg_linestyle(:), 'LineWidth', lw);
            xlabel('$L_1$','Interpreter','latex','FontSize', label_fs)
            ylabel('$L_2$','Interpreter','latex','FontSize', label_fs)
            legend('k-means + lambda','jensen','spg','FontSize', legend_fs)
%            legend('jensen','spg','spa','FontSize', legend_fs)
            hold off
            grid on
            grid minor
            ax = gca;
            ax.FontSize = axis_fs;
            
            figure
            colororder(newcolors)
            hold on
            title(sprintf('K=%d', Ks(idx_K)))
            plot(alphas, errstrain_rand{1}(:,idx_K,myrand),kmeans_lambda_linestyle(:), 'LineWidth', lw);
            plot(alphas, errstrain_rand{2}(:,idx_K,myrand),jensen_linestyle(:), 'LineWidth', lw);
            plot(alphas, errstrain_rand{3}(:,idx_K,myrand),spg_linestyle(:), 'LineWidth', lw);
            xlabel('$\alpha$','Interpreter','latex','FontSize', label_fs)
            ylabel('$err$','Interpreter','latex','FontSize', label_fs)
            legend('k-means + lambda','jensen','spg','FontSize', legend_fs)
%            legend('jensen','spg','spa','FontSize', legend_fs)
            set(gca,'xscale','log')
            hold off
            grid on

        end
        
    end
    
end

errstrain = cell(3,1);
scoretrain = cell(3,1);
errstest = cell(3,1);
scoretest = cell(3,1);
for i = 1:length(errstrain_rand)
    errstrain{i} = mean(errstrain_rand{i},3);
    scoretrain{i} = mean(scoretrain_rand{i},3);
    errstest{i} = mean(errstest_rand{i},3);
    scoretest{i} = mean(scoretest_rand{i},3);
end

if length(Ks) == 1
    idx_err = cell(3,1);
    idx_score = cell(3,1);
    for i = 1:length(errstest)
        [~,idx_err{i}] = min(errstest{i});
        [~,idx_score{i}] = max(scoretest{i});
    end
    
    figure
    subplot(2,2,1)
    colororder(newcolors)
    hold on
    plot(alphas,errstrain{1},kmeans_lambda_linestyle(:), 'LineWidth', lw)
    plot(alphas,errstrain{2},jensen_linestyle(:), 'LineWidth', lw)
    plot(alphas,errstrain{3},spg_linestyle(:), 'LineWidth', lw)
    xlabel('$\alpha$','interpreter','latex')
    ylabel('error on train')
    set(gca,'xscale','log')
    hold off
    
    subplot(2,2,2)
    colororder(newcolors)
    hold on
    plot(alphas,scoretrain{1},kmeans_lambda_linestyle(:), 'LineWidth', lw)
    plot(alphas,scoretrain{2},jensen_linestyle(:), 'LineWidth', lw)
    plot(alphas,scoretrain{3},spg_linestyle(:), 'LineWidth', lw)
    xlabel('$\alpha$','interpreter','latex')
    ylabel('fscore on train')
    set(gca,'xscale','log')
    hold off
    
    subplot(2,2,3)
    colororder(newcolors)
    hold on
    plot(alphas,errstest{1},kmeans_lambda_linestyle(:), 'LineWidth', lw)
    plot(alphas,errstest{2},jensen_linestyle(:), 'LineWidth', lw)
    plot(alphas,errstest{3},spg_linestyle(:), 'LineWidth', lw)
    xlabel('$\alpha$','interpreter','latex')
    ylabel('error on validation')
    set(gca,'xscale','log')
    hold off
    
    subplot(2,2,4)
    colororder(newcolors)
    hold on
    plot(alphas,scoretest{1},kmeans_lambda_linestyle(:), 'LineWidth', lw)
    plot(alphas,scoretest{2},jensen_linestyle(:), 'LineWidth', lw)
    plot(alphas,scoretest{3},spg_linestyle(:), 'LineWidth', lw)
    xlabel('$\alpha$','interpreter','latex')
    ylabel('fscore on validation')
    set(gca,'xscale','log')
    hold off
    
else
    [XX,YY] = meshgrid(alphas,Ks);
    
    figure
    surf(XX,YY,errstest')
    xlabel('$\alpha$','interpreter','latex')
    ylabel('$K$','interpreter','latex')
    zlabel('error on validation')
    
    figure
    surf(XX,YY,scoretest')
    xlabel('$\alpha$','interpreter','latex')
    ylabel('$K$','interpreter','latex')
    zlabel('f1score')
    
end

% compute final score on validation set
[idx_err1,idx_err2,~] = min_in_matrix(errstest);
K_sol = Ks(idx_err2);
alpha_sol = alphas(idx_err1);
[C_sol, Gamma_sol, PiX_sol, Lambda_sol, it, stats_train, L_out] = ...
    adamar_kmeans(Xdata', Ydata, K_sol, alpha_sol, maxIters, nrand);
C_sol = C_sol';

%[C_sol, Gamma_sol, PiX_sol, Lambda_sol, it, stats, L_out] = adamar_fmincon(Xdata', Ydata, K_sol, alpha_sol, maxIters);

Gammaval = compute_Gamma_kmeans(C_sol,Xval);
Yval_est = Lambda_sol*Gammaval;

err_sol = norm(Yval_est - Yval,'fro');
[stats] = statistics_multiclass(myonehotdecode(Yval_est,1:M,1), myonehotdecode(Yval,1:M,1));
score_sol = stats.f1score;

disp(['Performance on validation set:'])
disp(['K       = ' num2str(K_sol)])
disp(['alpha   = ' num2str(alpha_sol)])
disp(['err    = ' num2str(err_sol)])
disp(['score  = ' num2str(score_sol)])

