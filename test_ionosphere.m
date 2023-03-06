clear all
close all
addpath(genpath(pwd));

rng(42); %For reproducibility

VISUALIZE = true;
SVM = true;
CROSSVAL = false;
SPG = false;

%Fisher's Iris Data
load ionosphere
X(:,2) = [];

labels = categorical(Y);

%Order of the categories.
classes = categories(labels);

PiY = zeros(length(labels),1);
PiY(labels=='g') = 1;
T = size(X,1);

%[X, ~] = scaling(X, [], 'minmax');

tbl = array2table(X);
tbl.Y = PiY;
n = length(tbl.Y);


%(Hyper)parameters
maxIters = 100;
nrand = 5;
scaleT = true;
Ks = 25;
alphas = 0:0.1:1;
%alphas = 0.99:0.002:1;
test_size = 0.2;

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
            
            if SPG; [C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats_train, L_out] = adamar_fmincon(X, PiY, K, alpha, maxIters, nrand);
            else; [Lambda, C, Gamma, stats_train, L_out, PiX] = adamar_kmeans(X, PiY, K, alpha, maxIters, nrand, scaleT); end 
            lprecision(idx_alpha,idx_K) = stats_train.precision;
            lrecall(idx_alpha,idx_K) = stats_train.recall;
            lf1score(idx_alpha,idx_K, idx_fold) = stats_train.f1score;
            laccuracy(idx_alpha,idx_K) = stats_train.accuracy;
            
            %if stats_train.f1score > best_fscore{1}; best_fscore = {stats_train.f1score, alpha, K}; end
            
            Ls(idx_alpha,idx_K)  = L_out.L;
            L1s(idx_alpha,idx_K) = L_out.L1;
            L2s(idx_alpha,idx_K) = L_out.L2;
            
            if SPG
                [stats_test] = adamar_validate_ionosphere(Lambda, C, y, Piy);
            else
                [stats_test] = adamar_validate_ionosphere(Lambda, C', y, Piy); 
            end
            tprecision(idx_alpha,idx_K) = stats_test.precision;
            trecall(idx_alpha,idx_K) = stats_test.recall;
            tf1score(idx_alpha,idx_K, idx_fold) = stats_test.f1score;
            taccuracy(idx_alpha,idx_K) = stats_test.accuracy;
        end
end

if SVM
    SVMModel = fitcsvm(X, PiY');
    %SVMModel = fitcecoc(X, PiY, 'OptimizeHyperparameters', 'all');
    [labels_train,~] = predict(SVMModel,X);
    SVM_stats_train(idx_fold) = statistics(labels_train, PiY');
    [labels_test,~] = predict(SVMModel,y);
    SVM_stats_test(idx_fold) = statistics(labels_test, Piy');
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