clear all
close all
addpath(genpath(pwd));

rng(42); %For reproducibility

T = 1e2;
[X_true,Y_true,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem(T);

M = size(Y_true,1);
sigma = 0.015; % noise parameter
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

maxIters = 50;
nrand = 5;
Ks = [2,3,4,5,6]; %size(C_true,2);

epsilons = 10.^(-7:1:5);

Nrand = 10;

errs_rand = zeros(length(epsilons),length(Ks),Nrand);
score_rand = zeros(length(epsilons),length(Ks),Nrand);

for myrand = 1:Nrand
    idx = randperm(Ttrain+Ttest);
    Xperm = Xdata(:,idx);
    Yperm = Ydata(:,idx);
    
    Xtrain = Xperm(:,1:Ttrain);
    Xtest = Xperm(:,(Ttrain+1):end);
    Ytrain = Yperm(:,1:Ttrain);
    Ytest = Yperm(:,(Ttrain+1):end);
    
    for idx_epsilon=1:length(epsilons)
        epsilon = epsilons(idx_epsilon);
        for idx_K=1:length(Ks)
            K = Ks(idx_K);

            [C1, Gamma, PiX, Lambda, it1, stats_train, L_out] = ...
                adamar_kmeans(Xtrain', Ytrain, K, epsilon, maxIters, nrand);
            C = C1';
%           [C2, Gamma2, PiX2, Lambda2, it2, stats2, L_out2] = adamar_fmincon(X', Y, K, epsilon, maxIters, nrand);
%           [C3, Gamma3, PiX3, Lambda3, it3, stats3, L_out3] = adamar_spa(X', Y, K, epsilon, maxIters, nrand);
            
            
            
            % apply model to validation data
            Gammatest = compute_Gamma_kmeans(C,Xtest);
            Ytest_est = Lambda*Gammatest;
            
            errs_rand(idx_epsilon,idx_K,myrand) = norm(Ytest_est - Ytest,'fro');
            
            [stats] = statistics_multiclass(onehotdecode(Ytest_est,1:M,1), onehotdecode(Ytest,1:M,1));
            
            score_rand(idx_epsilon,idx_K,myrand) = stats.f1score;
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
    plot(epsilons,errs,'r')
    plot(epsilons(idx_err),errs(idx_err),'ro')
    xlabel('$\epsilon$','interpreter','latex')
    ylabel('error on validation')
    set(gca, 'XScale', 'log')
    hold off
    
    subplot(1,2,2)
    hold on
    plot(epsilons,score,'b')
    plot(epsilons(idx_score),score(idx_score),'bo')
    xlabel('$\epsilon$','interpreter','latex')
    ylabel('fscore on validation')
    set(gca, 'XScale', 'log')
    hold off
    
else
    [XX,YY] = meshgrid(epsilons,Ks);
    
    figure
    surf(XX,YY,errs')
    xlabel('$\epsilon$','interpreter','latex')
    ylabel('$K$','interpreter','latex')
    zlabel('error on validation')
    set(gca, 'XScale', 'log')

    figure
    surf(XX,YY,score')
    xlabel('$\epsilon$','interpreter','latex')
    ylabel('$K$','interpreter','latex')
    zlabel('f1score')   
    set(gca, 'XScale', 'log')
    
end

% compute final score on validation set
[idx_err1,idx_err2,~] = min_in_matrix(errs);
K_sol = Ks(idx_err2);
epsilon_sol = epsilons(idx_err1);
[C_sol, Gamma_sol, PiX_sol, Lambda_sol, it, stats_train, L_out] = ...
    adamar_kmeans(Xdata', Ydata, K_sol, epsilon_sol, maxIters, nrand);
C_sol = C_sol';

%[C_sol, Gamma_sol, PiX_sol, Lambda_sol, it, stats, L_out] = adamar_fmincon(Xdata', Ydata, K_sol, epsilon_sol, maxIters);

Gammaval = compute_Gamma_kmeans(C_sol,Xval);
Yval_est = Lambda_sol*Gammaval;
            
err_sol = norm(Yval_est - Yval,'fro');
[stats] = statistics_multiclass(onehotdecode(Yval_est,1:M,1), onehotdecode(Yval,1:M,1));
score_sol = stats.f1score;

disp(['Performance on validation set:'])
disp(['K       = ' num2str(K_sol)])
disp(['epsilon   = ' num2str(epsilon_sol)])
disp(['err    = ' num2str(err_sol)])
disp(['score  = ' num2str(score_sol)])

