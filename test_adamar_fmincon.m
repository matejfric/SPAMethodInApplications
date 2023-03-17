%ADAMAR FMINCON()
close all
clear all
addpath(genpath(pwd));

rng(42);

DATASET = 'DatasetSelection'; % 'Dataset', 'Dataset2', 'Dataset256', 'DatasetSelection'
VISUALIZE = false;

[X, ca_Y] = get_train_test_data(DATASET);

%[X, ca_Y] = correlation_analysis(X, ca_Y); % Removal of strongly correlated columns

PiY = [X(:,end), 1-X(:,end)]';
X = X(:,1:end-1);

%[X, ca_Y] = scaling(X, ca_Y, 'minmax');
[X, ca_Y] = scaling(X, ca_Y, 'zscore', 'robust');
%Standardizing is usually done when the variables on which the PCA is performed are not measured on the same scale. Note that standardizing implies assigning equal importance to all variables.
[X, ca_Y] = mypca(X, ca_Y);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%ADAMAR
%alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1-1e-1, 1-1e-2, 1-1e-3];
%alphas = 0.001:0.0003:0.002;
alphas = 0:0.25:1;
alphas = [0.25, 0.75];
Ks = 25; % Number of clusters
maxIters = 25;
nrand = 3;

for a = 1:numel(alphas)
    alpha = alphas(a);
    
    for k = 1 : length(Ks)
        K = Ks(k);
        [C, Gamma, PiX, Lambda, it, stats_train, L_out] = ...
            adamar_fmincon(X, PiY, K, alpha, maxIters, nrand);
        
        % Display transition matrix
        disp(Lambda);
        
        % Save objective function value for L-curve
        Ls(a,k) = L_out.L;
        L1s(a,k) = L_out.L1;
        L2s(a,k) = L_out.L2;
        
        lprecision(a,k) = stats_train.precision;
        lrecall(a,k) = stats_train.recall;
        lf1score(a,k) = stats_train.f1score;
        laccuracy(a,k) = stats_train.accuracy;

        disp("Lambda:"); disp(Lambda); % Transition matrix
        [stats_test] = adamar_predict(Lambda, C, Ks(k), alphas(a), ca_Y, DATASET, VISUALIZE);
        tprecision(a,k) = stats_test.precision;
        trecall(a,k) = stats_test.recall;
        tf1score(a,k) = stats_test.f1score;
        taccuracy(a,k) = stats_test.accuracy;
        
        %Score plot
        if VISUALIZE; score_plot(sprintf('Adamar SPG, K=%d, alpha=%.2e', K, alpha), K, lprecision(a,:,end), lrecall(a,:,end), lf1score(a,:,end), laccuracy(a,:,end), tprecision(a,:), trecall(a,:), tf1score(a,:), taccuracy(a,:)); end
    end
end

regularization_plot(sprintf('Adamar SPG, k=%d', Ks), alphas,lprecision, lrecall, lf1score, laccuracy,tprecision, trecall, tf1score, taccuracy)

plot_L_curves(Ls, L1s, L2s, Ks, alphas);

