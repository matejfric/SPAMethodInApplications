clear all
close all
addpath(genpath(pwd));

rng(42); %For reproducibility

T = 1e2;
[X_true,Y_true,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem(T);

M = size(Y_true,1);
sigma = 0.15; % noise parameter
X = X_true + sigma * randn(size(X_true));

Y = Y_true;
nwrong = floor(0.05*T); % wrong labels
idx = randperm(T);
for i=1:nwrong
  Y(1:2,idx(i)) = Y(2:-1:1,idx(i));
end

Ttrain = floor(0.6*T);
Ttest = floor(0.2*T);
Tval = floor(0.2*T);

Xdata = X(:,1:Ttrain+Ttest);
Ydata = Y(:,1:Ttrain+Ttest);

Xval = X(:,(Ttrain+Ttest+1):end);
Yval = Y(:,(Ttrain+Ttest+1):end);

maxIters = 30;
nrand = 10;
Ks = [2,3,4,5,6]; %size(C_true,2);

%alphas = 0:0.005:0.02;
alphas = 0:0.2:1;
%alphas = 0.05:0.05:0.95;
%alphas = 0.9:0.01:0.99;

Nrand = 10;

errs_rand = zeros(length(alphas),length(Ks),Nrand);
score_rand = zeros(length(alphas),length(Ks),Nrand);

for myrand = 1:Nrand
    idx = randperm(Ttrain+Ttest);
    Xperm = Xdata(:,idx);
    Yperm = Ydata(:,idx);
    
    Xtrain = Xperm(:,1:Ttrain);
    Xtest = Xperm(:,(Ttrain+1):end);
    Ytrain = Yperm(:,1:Ttrain);
    Ytest = Yperm(:,(Ttrain+1):end);
    
    for idx_alpha=1:length(alphas)
        alpha = alphas(idx_alpha);
        for idx_K=1:length(Ks)
            K = Ks(idx_K);

            [C1, Gamma, PiX, Lambda, it1, stats_train, L_out] = ...
                adamar_kmeans(Xtrain', Ytrain, K, alpha, maxIters, nrand);
            C = C1';
%           [C2, Gamma2, PiX2, Lambda2, it2, stats2, L_out2] = adamar_fmincon(X', Y, K, alpha, maxIters, nrand);
%           [C3, Gamma3, PiX3, Lambda3, it3, stats3, L_out3] = adamar_spa(X', Y, K, alpha, maxIters, nrand);
            
            
            
            % apply model to validation data
            Gammatest = compute_Gamma_kmeans(C,Xtest);
            Ytest_est = Lambda*Gammatest;
            
            errs_rand(idx_alpha,idx_K,myrand) = norm(Ytest_est - Ytest,'fro');
            
            [stats] = statistics_multiclass(onehotdecode(Ytest_est,1:M,1), onehotdecode(Ytest,1:M,1));
            
            score_rand(idx_alpha,idx_K,myrand) = stats.f1score;
        end
    end
    
end

errs = mean(errs_rand,3);
score = mean(score_rand,3);

if length(Ks) == 1
    [~,idx_err] = min(errs);
    [~,idx_score] = max(score);
    
    figure
    subplot(1,2,1)
    hold on
    plot(alphas,errs,'r')
    plot(alphas(idx_err),errs(idx_err),'ro')
    xlabel('$\alpha$','interpreter','latex')
    ylabel('error on validation')
    hold off
    
    subplot(1,2,2)
    hold on
    plot(alphas,score,'b')
    plot(alphas(idx_score),score(idx_score),'bo')
    xlabel('$\alpha$','interpreter','latex')
    ylabel('fscore on validation')
    hold off
    
else
    [XX,YY] = meshgrid(alphas,Ks);
    
    figure
    surf(XX,YY,errs')
    xlabel('$\alpha$','interpreter','latex')
    ylabel('$K$','interpreter','latex')
    zlabel('error on validation')

    figure
    surf(XX,YY,score')
    xlabel('$\alpha$','interpreter','latex')
    ylabel('$K$','interpreter','latex')
    zlabel('f1score')    
    
end

% compute final score on validation set
[idx_err1,idx_err2,~] = min_in_matrix(errs);
K_sol = Ks(idx_err2);
alpha_sol = alphas(idx_err1);
[C_sol, Gamma_sol, PiX_sol, Lambda_sol, it, stats_train, L_out] = ...
    adamar_kmeans(Xdata', Ydata, K_sol, alpha_sol, maxIters, nrand);
C_sol = C_sol';

%[C_sol, Gamma_sol, PiX_sol, Lambda_sol, it, stats, L_out] = adamar_fmincon(Xdata', Ydata, K_sol, alpha_sol, maxIters);

Gammaval = compute_Gamma_kmeans(C_sol,Xval);
Yval_est = Lambda_sol*Gammaval;
            
err_sol = norm(Yval_est - Yval,'fro');
[stats] = statistics_multiclass(onehotdecode(Yval_est,1:M,1), onehotdecode(Yval,1:M,1));
score_sol = stats.f1score;

disp(['Performance on validation set:'])
disp(['K       = ' num2str(K_sol)])
disp(['alpha   = ' num2str(alpha_sol)])
disp(['err    = ' num2str(err_sol)])
disp(['score  = ' num2str(score_sol)])

