%close all
clear all
addpath(genpath(pwd));
rng(42);
MRMR = [];
PCA = [];
SCALE = [];
descriptorsEnum = [];

[X,y] = get_iris_data();
%[X,y] = get_bcwd_data();

%[X, ~, ~, SCALE] = scaling(X, [], 'minmax');
%[X, ~, ~, SCALE] = scaling(X, [], 'zscore', 'robust');

% Rank features for classification using minimum redundancy maximum relevance (MRMR) algorithm
% [idx,mrmr_scores] = fscmrmr(X,y);
% figure
% bar(mrmr_scores(idx))
% xlabel('Predictor rank')
% ylabel('Predictor importance score')
% MRMR.scores = mrmr_scores;
% MRMR.limit = 0.15;
% X = X(:, mrmr_scores > MRMR.limit);

% % Univariate feature ranking for classification using chi-square tests
% [idx,scores] = fscchi2(X,y);
% figure
% bar(scores(idx))
% xlabel('Predictor rank')
% ylabel('Predictor importance score')

%[X, ca_Y, PCA] = mypca(X, [], y);

[X_train, X_val, X_test, y_train, y_val, y_test] = train_test_val_split(X,y,0.6,0.2);

test_standard_classifiers(X_train,X_test,y_train,y_test);

% Train KKLDJ
%mdl = KKLDJ();
%mdl = KKLD();
%mdl = KLambda();
mdl = SPAKLD();

mdl.MRMR = MRMR;
mdl.PCA = PCA;
mdl.SCALE = SCALE;
mdl.Nrand = 15;
mdl.descriptors = descriptorsEnum;
mdl.verbose = false;
epsilons = 10.^(-7:1:5);
%Ks = [5,10,15];
%Ks = [10,20,30,40,50];
%Ks = [25,30,35,40];
Ks = 3;
%Ks = 2:2:20;

Ls  = zeros(numel(epsilons),length(Ks));
L1s = zeros(numel(epsilons),length(Ks));
L2s = zeros(numel(epsilons),length(Ks));

% Training/validation
fprintf("Training...\n");
for a = progress(1:length(epsilons))
    for k = 1:length(Ks)
        mdl.epsilon = epsilons(a);
        mdl.K = Ks(k);
        mdl.fit(X_train,y_train);
        
        Ls(a,k)  = mdl.L.L;
        L1s(a,k) = mdl.L.L1;
        L2s(a,k) = mdl.L.L2;
        
        [~,y_pred] = mdl.predict(X_val');
        stats_val(a,k)  = mdl.computeStats(y_pred, y_val');
        stats_train(a,k) = mdl.statsTrain;
    end
end
mdl.statsTrain = stats_train;
mdl.printStatsTrain();

for k = 1:length(Ks)
    plot_L_curves(Ls(:,k), L1s(:,k), L2s(:,k), epsilons, Ks(k));
    plot_f1score(stats_train(:,k), stats_val(:,k), epsilons, Ks(k));
end

% Grid search
if sum([size(stats_val)] >= [2,2]) >= 2
    [best_epsilon, best_K, best_score] = grid_search(stats_val,epsilons,Ks);
else
    if length(Ks) > length(epsilons)
        [best_score, k] = max([stats_val.f1score]);
        best_K = Ks(k);
        best_epsilon = epsilons;
    else
        [best_score, a] = max([stats_val.f1score]);
        best_epsilon = epsilons(a);
        best_K = Ks;
    end
end

% Test
mdl.epsilon = best_epsilon;
mdl.K = best_K;
mdl.fit(X_train,y_train);
[~,y_pred] = mdl.predict(X_test');
stats_test = mdl.computeStats(y_pred, y_test');
fprintf("\nTest performance:\nF1-score = %.3f\nAccuracy = %.3f\n\n",...
    stats_test.f1score, stats_test.accuracy);

if mdl.mdl.classes <= 2 
    % Find the threshold that maximizes the balance
    % between sensitivity and specificity.

    % Calculate ROC curve and AUC
    [TPR,FPR,T,AUC,OPTROCPT] = perfcurve(y_test',y_pred,1);

    % Plot ROC curve
    figure
    plot(TPR,FPR, 'DisplayName', 'ROC curve', 'Linewidth', 2)
    hold on
    plot(OPTROCPT(1),OPTROCPT(2),'ro', 'DisplayName', 'Optimal ROC operating point', 'Linewidth', 2)
    xlabel('False positive rate (FPR)', 'FontSize',13, 'Interpreter', 'latex')
    ylabel('True positive rate (TPR)', 'FontSize',13, 'Interpreter', 'latex')
    title(['ROC curve (AUC = ' num2str(AUC,3) ')'], 'Interpreter', 'latex', 'FontSize',15)
    legend('show', 'Location', 'southeast', 'Interpreter', 'latex', 'FontSize',14)

    % Choose threshold (Youden's J statistic)
    J = max(abs(TPR-FPR));
    threshold = T(find(abs(TPR-FPR)==J,1));

    % Classify new data using threshold
    y_pred_J = mdl.predict(X_test') >= threshold;
    stats_test = mdl.computeStats(double(y_pred_J), y_test');

    fprintf("\nTest performance (Youden's J statistic):\nF1-score = %.3f\nAccuracy = %.3f\n\n",...
        stats_test.f1score, stats_test.accuracy);
end

%Save the pre-trained model (object of class KKLDJ) to a .mat file
% save('ModelKKLDJ.mat', 'mdl', '-v7.3'); 
%Load the pre-trained model 
% my_mdl = load('ModelKKLDJ.mat').mdl;


