%ADAMAR FMINCON()
close all
clear all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/Adamar')
addpath('ProgramFiles/SPG')

rng(42);

DATASET = 'Dataset256'; % 'Dataset', 'Dataset2', 'Dataset256'
VISUALIZE = false;

[X, ca_Y] = get_train_test_data(DATASET);

% Removal of strongly correlated columns
[X, ca_Y] = correlation_analysis(X, ca_Y);

% Scaling
%[X, ca_Y] = scaling(X, ca_Y, 'minmax');

% PCA
%[X, ca_Y] = principal_component_analysis(X, ca_Y);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

%ADAMAR
PiY = [X(:,end), 1-X(:,end)]';
X = X(:,1:end-1);
%alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1-1e-1, 1-1e-2, 1-1e-3];
alphas = 0.01:0.003:0.02;
Ks = 25; % Number of clusters
maxIters = 10;
nrand = 3;

for a = 1:numel(alphas)
    alpha = alphas(a);
    
    for k = 1 : length(Ks)
        K = Ks(k);
        
        [C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats_train, L_out] = ...
            adamar_fmincon(X, PiY, K, alpha, maxIters, nrand);
        
        % Display transition matrix
        disp(Lambda);
        
        % Save objective function value for L-curve
        Ls(a,k) = L_out.L;
        L1s(a,k) = L_out.L1;
        L2s(a,k) = L_out.L2;

         for i = 1:maxIters
            if i <= it
            lprecision(a,k,i) = stats_train(i).precision;
            lrecall(a,k,i) = stats_train(i).recall;
            lf1score(a,k,i) = stats_train(i).f1score;
            laccuracy(a,k,i) = stats_train(i).accuracy;
            else
                lprecision(a,k,i) = NaN;
                lrecall(a,k,i) = NaN;
                lf1score(a,k,i) = NaN;
                laccuracy(a,k,i) = NaN;
            end
        end

        disp("Lambda:"); disp(Lambda); % Transition matrix

        [stats_test] = adamar_predict_mat(Lambda, C, Ks(k), alphas(a), [], [], ca_Y, DATASET, VISUALIZE);
        tprecision(a,k) = stats_test.precision;
        trecall(a,k) = stats_test.recall;
        tf1score(a,k) = stats_test.f1score;
        taccuracy(a,k) = stats_test.accuracy;
        
        %Score plot
        if VISUALIZE; score_plot(sprintf('Adamar SPG, K=%d, alpha=%.2e', K, alpha), K, lprecision(a,:,end), lrecall(a,:,end), lf1score(a,:,end), laccuracy(a,:,end), tprecision(a,:), trecall(a,:), tf1score(a,:), taccuracy(a,:)); end
    end
end

plot_L_curves(Ls, L1s, L2s, Ks, alphas);

