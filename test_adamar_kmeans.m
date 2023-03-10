%ADAMAR K-MEANS
%close all
clear all
addpath(genpath(pwd));

rng(42);

DATASET = 'Dataset256'; % 'Dataset', 'Dataset2', 'Dataset256'
VISUALIZE = false;

[X, ca_Y] = get_train_test_data(DATASET);

% Removal of strongly correlated columns
[X, ca_Y] = correlation_analysis(X, ca_Y);

PiY = [X(:,end), 1-X(:,end)]; % Ground truth
X = X(:,1:end-1);

% Scaling
[X, ca_Y] = scaling(X, ca_Y, 'minmax');

% PCA
%[X, ca_Y] = principal_component_analysis(X, ca_Y);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%Ks = [2,5,12];
Ks = 25;
%Ks = 100;

maxIters = 100;

%alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5];
%alphas = 0:0.1:1;
alphas = 0.8:0.03:1;

L1s = zeros(numel(alphas),length(Ks));
L2s = zeros(numel(alphas),length(Ks));

nrand = 3; % Number of random runs (annealing)

for a = 1:numel(alphas)
    for k = 1 : length(Ks)
        [Lambda, C, Gamma, stats_train, L_out, PiX] = adamar_kmeans(X, PiY', Ks(k), alphas(a), maxIters, nrand);
        lprecision(a,k) = stats_train.precision;
        lrecall(a,k) = stats_train.recall;
        lf1score(a,k) = stats_train.f1score;
        laccuracy(a,k) = stats_train.accuracy;

        Ls(a,k) = L_out.L;
        L1s(a,k) = L_out.L1;
        L2s(a,k) = L_out.L2;

        disp("Lambda:"); disp(Lambda); % Transition matrix
        
        [stats_test] = adamar_predict_mat(Lambda, C', Ks(k), alphas(a), [], [], ca_Y, DATASET, VISUALIZE);
        tprecision(a,k) = stats_test.precision;
        trecall(a,k) = stats_test.recall;
        tf1score(a,k) = stats_test.f1score;
        taccuracy(a,k) = stats_test.accuracy;
    end

    %score_plot(sprintf('Adamar k-means, K=%d, alpha=%.2e', K, alpha(a)), K, lprecision(a,:), lrecall(a,:), lf1score(a,:), laccuracy(a,:), tprecision(a,:), trecall(a,:), tf1score(a,:), taccuracy(a,:))

end

regularization_plot(sprintf('Adamar K-means, k=%d', Ks), alphas,lprecision, lrecall, lf1score, laccuracy,tprecision, trecall, tf1score, taccuracy)

plot_L_curves(Ls, L1s, L2s, Ks, alphas);

fprintf("\nProgram finished successfully.\n");

