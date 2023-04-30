%ADAMAR K-MEANS
%close all
clear all
addpath(genpath(pwd));

rng(42);

DATASET = 'DatasetSelection'; % 'Dataset', 'Dataset2', 'Dataset256', 'DatasetSelection'
VISUALIZE = false;

[X, ca_Y] = get_train_test_data(DATASET, 0.8);

% Removal of strongly correlated columns
%[X, ca_Y] = correlation_analysis(X, ca_Y);

%[X, ca_Y] = my_umap();

PiY = [X(:,end), 1-X(:,end)]; % Ground truth
X = X(:,1:end-1);

% Scaling
[X, ca_Y] = scaling(X, ca_Y, 'minmax');
%[X, ca_Y] = scaling(X, ca_Y, 'zscore', 'robust');

% PCA
[X, ca_Y, explained] = mypca(X, ca_Y);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ", sum(PiY(:,1)), sum(PiY(:,2)));


%Ks = [2,5,12];
%Ks = 25;
%Ks = [2,4,8,16,32,64];
Ks = 25;


maxIters = 1000;

%alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5];
%alphas = 0:0.1:1;
%alphas = 0.9:0.01:1;
%alphas = 0.99:0.001:1;
%alphas = 0.999:0.0002:1;
%alphas = 0.9999:0.00002:1; % tady se ten algoritmus chová hezky
%alphas = 0.99999:0.000002:1; % až 83% f1-skóre na testovací mže
%alphas = 0.999999:0.0000002:1; % tady už nic moc, možná už to naráží na numerické chyby?
alphas = 0.99996;

L1s = zeros(numel(alphas),length(Ks));
L2s = zeros(numel(alphas),length(Ks));

nrand = 3; % Number of random runs (annealing)

for a = 1:numel(alphas)
    for k = 1 : length(Ks)
        [C, Gamma, PiX, Lambda, it, stats_train, L_out] = ...
            adamar_kmeans(X, PiY', Ks(k), alphas(a), maxIters, nrand);
        lprecision(a,k) = stats_train.precision;
        lrecall(a,k) = stats_train.recall;
        lf1score(a,k) = stats_train.f1score;
        laccuracy(a,k) = stats_train.accuracy;

        Ls(a,k) = L_out.L;
        L1s(a,k) = L_out.L1;
        L2s(a,k) = L_out.L2;

        disp("Lambda:"); disp(Lambda); % Transition matrix
        [stats_test] = adamar_predict(Lambda, C', Ks(k), alphas(a), ca_Y, DATASET, VISUALIZE);
        tprecision(a,k) = stats_test.precision;
        trecall(a,k) = stats_test.recall;
        tf1score(a,k) = stats_test.f1score;
        taccuracy(a,k) = stats_test.accuracy;
    end

    %score_plot(sprintf('Adamar k-means, K=%d, alpha=%.2e', K, alpha(a)), K, lprecision(a,:), lrecall(a,:), lf1score(a,:), laccuracy(a,:), tprecision(a,:), trecall(a,:), tf1score(a,:), taccuracy(a,:))

end

% Ks plot
if false
    figure 
    subplot(1,2,1)
    hold on
    plot(Ks, laccuracy, 'o-' , 'LineWidth', 2);
    plot(Ks, lf1score, 'o-' , 'LineWidth', 2);
    hold off
    subplot(1,2,2)
    hold on
    plot(Ks, tf1score, 'o-' , 'LineWidth', 2);
    plot(Ks, taccuracy, 'o-' , 'LineWidth', 2);
    hold off
    legend('F1-score', 'Accuracy', 'Location','southeast')
end

plot_L_curves(Ls, L1s, L2s, Ks, alphas);

%regularization_plot(sprintf('Adamar K-means, k=%d', Ks), alphas,lprecision, lrecall, lf1score, laccuracy,tprecision, trecall, tf1score, taccuracy)

fprintf("\nProgram finished successfully.\n");

