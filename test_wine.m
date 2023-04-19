% https://archive.ics.uci.edu/ml/datasets/Wine
% https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
%
% Inspired by:
% https://www.kaggle.com/code/vishalyo990/prediction-of-quality-of-wine
%
% These data are the results of a chemical analysis of wines
% grown in the same region in Italy but derived from three
% different cultivars. The analysis determined the quantities
% of 13 constituents found in each of the three types of wines.
%
% In a classification context, this is a well posed problem
% with "well behaved" class structures. A good data set
% for first testing of a new classifier, but not very challenging.

clear all
close all
addpath(genpath(pwd));

rng(42); %For reproducibility

VISUALIZE = true;
SVM = true;
NB = true;
CROSSVAL = false;
SPG = false;

% Wine Dataset
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
DS = readtable('winequality-red.csv');

% Transformation of the target variable for binary classification
[quality] = discretize(DS.quality,[2, 6.5, 8]); 

%labels = categorical(DS.quality);
labels = categorical(quality);
classes = categories(labels);
PiY = onehotencode(labels,2);

[X, ~] = scaling(table2array(DS(:,1:end-1)), [], 'zscore', 'robust');
%[X, ~] = scaling(table2array(DS(:,1:end-1)), [], 'minmax');

tbl = array2table(X);
tbl.Y = PiY;
n = length(tbl.Y);

%(Hyper)parameters
maxIters = 100;
nrand = 5;
scaleT = true;
Ks = 100;
alphas = 0:0.1:1;
%alphas = 0.95:0.01:1;
%alphas = 0.99:0.001:1;
%alphas = 0.999:0.0001:1;
%alphas = 0.9996;
%alphas = 0.9999:0.00001:1;
test_size = 0.20;

if CROSSVAL
    KFold = 10;
    partition = cvpartition(labels,'KFold',KFold,'Stratify',true);
else
    KFold = 1;
    partition = cvpartition(n, 'Holdout', test_size); 
end

%Cross-Validation
for idx_fold = 1:KFold
idxTrain = training(partition,idx_fold);
tblTrain = tbl(idxTrain,:);
idxTest = test(partition,idx_fold);
tblTest = tbl(idxTest,:);

%Train
X = table2array(tblTrain(:,1:end-1));
PiY = tblTrain.Y';

%Test
y = table2array(tblTest(:,1:end-1));
Piy = tblTest.Y';

%Preallocation
Ls  = zeros(numel(alphas),length(Ks));
L1s = zeros(numel(alphas),length(Ks));
L2s = zeros(numel(alphas),length(Ks));
best_fscore = {0,0,0};

for idx_alpha=1:length(alphas)
    alpha = alphas(idx_alpha);
    
        for idx_K=1:length(Ks)
            K = Ks(idx_K);
            
            if SPG
                [C, Gamma, PiX, Lambda, it, stats_train, L_out] = ...
                    adamar_fmincon(X, PiY, K, alpha, maxIters, nrand);
            else
                [C, Gamma, PiX, Lambda, it, stats_train, L_out] = ...
                    adamar_kmeans(X, PiY, K, alpha, maxIters, nrand, scaleT); 
            end
            lprecision(idx_alpha,idx_K) = stats_train.precision;
            lrecall(idx_alpha,idx_K) = stats_train.recall;
            lf1score(idx_alpha,idx_K, idx_fold) = stats_train.f1score;
            laccuracy(idx_alpha,idx_K) = stats_train.accuracy;
            
            %if stats_train.f1score > best_fscore{1}; best_fscore = {stats_train.f1score, alpha, K}; end
            
            Ls(idx_alpha,idx_K)  = L_out.L;
            L1s(idx_alpha,idx_K) = L_out.L1;
            L2s(idx_alpha,idx_K) = L_out.L2;
            
            if SPG; [stats_test] = adamar_validate_fisher_iris(Lambda, C, y, Piy, classes);
            else [stats_test] = adamar_validate_fisher_iris(Lambda, C', y, Piy, classes); end
            tprecision(idx_alpha,idx_K) = stats_test.precision;
            trecall(idx_alpha,idx_K) = stats_test.recall;
            tf1score(idx_alpha,idx_K, idx_fold) = stats_test.f1score;
            taccuracy(idx_alpha,idx_K) = stats_test.accuracy;
        end
end

if SVM
    template = templateSVM('KernelFunction', 'gaussian', 'PolynomialOrder', [], ...
        'KernelScale', 0.33289, 'BoxConstraint', 10.868, 'Standardize', true);
    SVMModel = fitcecoc(X, onehotdecode(PiY',classes,2), 'Learners', template, 'Coding', 'onevsall');
    %[SVMModel,HyperparameterOptimizationResults] = fitcecoc(X, onehotdecode(PiY',classes,2), 'OptimizeHyperparameters', 'all');

    [labels_train,~] = predict(SVMModel,X);
    SVM_stats_train(idx_fold) = statistics(labels_train, onehotdecode(PiY',classes,2));
    [labels_test,~] = predict(SVMModel,y);
    SVM_stats_test(idx_fold) = statistics(labels_test, onehotdecode(Piy',classes,2));
end

if NB
    NBModel = fitcnb(X, onehotdecode(PiY',classes,2));
    [labels_train,~] = predict(NBModel,X);
    NB_stats_train(idx_fold) = statistics(labels_train, onehotdecode(PiY',classes,2));
    [labels_test,~] = predict(NBModel,y);
    NB_stats_test(idx_fold) = statistics(labels_test, onehotdecode(Piy',classes,2));
end

if VISUALIZE
    if SPG
        title = sprintf('Adamar SPG, K=%d', K);
    else
        title = sprintf('Adamar K-means, K=%d', K);
    end
    score_plot(title,...
         alphas, lprecision(:,idx_K), lrecall(:,idx_K), lf1score(:,idx_K), laccuracy(:,idx_K),...
         tprecision(:,idx_K), trecall(:,idx_K), tf1score(:,idx_K), taccuracy(:,idx_K));
    plot_L_curves(Ls, L1s, L2s, Ks, alphas, title);
end

end

fprintf("Adamar -> Mean learning F1-score: %.2f | Mean testing F1-score: %.2f\n", mean(lf1score), mean(tf1score));
if SVM
    fprintf("SVM ----> Mean learning F1-score: %.2f | Mean SVM testing F1-score: %.2f\n",mean([SVM_stats_train.f1score]), mean([SVM_stats_test.f1score]));
end
if SVM
    fprintf("NB -----> Mean learning F1-score: %.2f | Mean SVM testing F1-score: %.2f\n",mean([NB_stats_train.f1score]), mean([NB_stats_test.f1score]));
end