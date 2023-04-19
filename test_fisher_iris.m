clear all
%close all
addpath(genpath(pwd));

rng(42); %For reproducibility

VISUALIZE = true;
SVM = true;
CROSSVAL = false;
SPG = false;

%Fisher's Iris Data
load fisheriris

labels = categorical(species);

%Order of the categories.
classes = categories(labels);

%Encode the labels into one-hot vectors.
%Expand the labels into vectors in the second dimension to encode the classes.
PiY = onehotencode(labels,2);
T = size(meas,1);

X = meas;
%[X, ~] = scaling(X, [], 'zscore-robust');
%[X, ~] = scaling(X, [], 'zscore');
%[X, ~] = scaling(X, [], 'minmax');


tbl = array2table(X);
tbl.Y = PiY;
n = length(tbl.Y);


%(Hyper)parameters
maxIters = 100;
nrand = 5;
scaleT = true;
Ks = 3;
epsilons = 10.^(-7:1:5);
test_size = 0.5;

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
X = table2array(tblTrain(:,1:4));
PiY = tblTrain.Y';

%Test
y = table2array(tblTest(:,1:4));
Piy = tblTest.Y';

%Preallocation
Ls  = zeros(numel(epsilons),length(Ks));
L1s = zeros(numel(epsilons),length(Ks));
L2s = zeros(numel(epsilons),length(Ks));
best_fscore = {0,0,0};

for idx_epsilon=1:length(epsilons)
    epsilon = epsilons(idx_epsilon);
    
        for idx_K=1:length(Ks)
            K = Ks(idx_K);
            
            if SPG
                %[C, Gamma, PiX, Lambda, it, stats_train, L_out] = ...
                %    adamar_fmincon(X, PiY, K, epsilon, maxIters, nrand);
                [C, Gamma, PiX, Lambda, it, stats_train, L_out] = ...
                    adamar_spa(X, PiY, K, epsilon, maxIters, nrand);
            else
                [C, Gamma, PiX, Lambda, it, stats_train, L_out] = ...
                    adamar_kmeans(X, PiY, K, epsilon, maxIters, nrand, scaleT);
                %[C, Gamma, PiX, Lambda, it, stats_train, L_out] = ...
                %    kmeans_lambda(X, PiY, K, epsilon);
            end
            lprecision(idx_epsilon,idx_K) = stats_train.precision;
            lrecall(idx_epsilon,idx_K) = stats_train.recall;
            lf1score(idx_epsilon,idx_K, idx_fold) = stats_train.f1score;
            laccuracy(idx_epsilon,idx_K) = stats_train.accuracy;
            
            Ls(idx_epsilon,idx_K)  = L_out.L;
            L1s(idx_epsilon,idx_K) = L_out.L1;
            L2s(idx_epsilon,idx_K) = L_out.L2;
            
            if SPG; [stats_test] = adamar_validate_fisher_iris(Lambda, C, y, Piy, classes);
            else [stats_test] = adamar_validate_fisher_iris(Lambda, C', y, Piy, classes); end
            tprecision(idx_epsilon,idx_K) = stats_test.precision;
            trecall(idx_epsilon,idx_K) = stats_test.recall;
            tf1score(idx_epsilon,idx_K, idx_fold) = stats_test.f1score;
            taccuracy(idx_epsilon,idx_K) = stats_test.accuracy;
        end
end

if SVM
    SVMModel = fitcecoc(X, onehotdecode(PiY',classes,2));
    %SVMModel = fitcecoc(X, onehotdecode(PiY',classes,2), 'OptimizeHyperparameters', 'all');
    [labels_train,~] = predict(SVMModel,X);
    SVM_stats_train(idx_fold) = statistics_multiclass(labels_train, onehotdecode(PiY',classes,2));
    [labels_test,~] = predict(SVMModel,y);
    SVM_stats_test(idx_fold) = statistics_multiclass(labels_test, onehotdecode(Piy',classes,2));
end

if VISUALIZE
    if SPG
        title = sprintf('Adamar SPG, K=%d', K);
    else
        title = sprintf('Adamar K-means, K=%d', K);
    end
    score_plot(title,...
         epsilons, lprecision(:,idx_K), lrecall(:,idx_K), lf1score(:,idx_K), laccuracy(:,idx_K),...
         tprecision(:,idx_K), trecall(:,idx_K), tf1score(:,idx_K), taccuracy(:,idx_K));
    plot_L_curves(Ls, L1s, L2s, epsilons, Ks, title);
end

end

fprintf("Adamar -> Mean learning F1-score: %.2f | Mean testing F1-score: %.2f\n", mean(lf1score), mean(tf1score));
if SVM
    fprintf("SVM ----> Mean learning F1-score: %.2f | Mean SVM testing F1-score: %.2f\n",mean([SVM_stats_train.f1score]), mean([SVM_stats_test.f1score]));
end
