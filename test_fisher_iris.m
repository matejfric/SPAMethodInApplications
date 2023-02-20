clear all
close all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans
addpath('ProgramFiles/SPG') 

rng(42); %For reproducibility

VISUALIZE = true;
SVM = true;
CROSSVAL = false;

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
[X, ~] = scaling(X, [], 'minmax');

tbl = array2table(X);
tbl.Y = PiY;
n = length(tbl.Y);

if CROSSVAL
    KFold = 10;
    partition = cvpartition(labels,'KFold',KFold,'Stratify',true);
else
    KFold = 1;
    test_size = 0.25;
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

%(Hyper)parameters
maxIters = 30;
nrand = 10;
scaleT = true;
Ks = 3;
alphas = 0:0.05:1;

%Preallocation
Ls  = zeros(numel(alphas),length(Ks));
L1s = zeros(numel(alphas),length(Ks));
L2s = zeros(numel(alphas),length(Ks));
best_fscore = {0,0,0};

for idx_alpha=1:length(alphas)
    alpha = alphas(idx_alpha);
    
        for idx_K=1:length(Ks)
            K = Ks(idx_K);
    
            [Lambda, C, Gamma, stats_train, L_out, PiX] = adamar_kmeans(X, PiY, K, alpha, maxIters, nrand, scaleT);
            %[C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats, L_out] = adamar_fmincon(X', PiY, K, alpha, maxIters);
            lprecision(idx_alpha,idx_K) = stats_train.precision;
            lrecall(idx_alpha,idx_K) = stats_train.recall;
            lf1score(idx_alpha,idx_K, idx_fold) = stats_train.f1score;
            laccuracy(idx_alpha,idx_K) = stats_train.accuracy;
            
            if stats_train.f1score > best_fscore{1}
                best_fscore = {stats_train.f1score, alpha, K};
            end
            
            Ls(idx_alpha,idx_K)  = L_out.L;
            L1s(idx_alpha,idx_K) = L_out.L1;
            L2s(idx_alpha,idx_K) = L_out.L2;
            
            [stats_test] = adamar_validate_fisher_iris(Lambda, C', y, Piy, classes);
            tprecision(idx_alpha,idx_K) = stats_test.precision;
            trecall(idx_alpha,idx_K) = stats_test.recall;
            tf1score(idx_alpha,idx_K, idx_fold) = stats_test.f1score;
            taccuracy(idx_alpha,idx_K) = stats_test.accuracy;
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

score_plot(sprintf('Adamar k-means, K=%d', K),...
     alphas, lprecision(:,idx_K), lrecall(:,idx_K), lf1score(:,idx_K), laccuracy(:,idx_K),...
     tprecision(:,idx_K), trecall(:,idx_K), tf1score(:,idx_K), taccuracy(:,idx_K))

% L-curve
for idx_K=1:length(Ks)
    figure
    hold on
    title(sprintf('K=%d', K))
    plot(L1s(:,idx_K), L2s(:,idx_K),'r-o');
    %text(L1s(1),L2s(1),['$\alpha = ' num2str(alpha(1)) '$'],'Interpreter','latex')
    %text(L1s(end),L2s(end),['$\alpha = ' num2str(alpha(end)) '$'],'Interpreter','latex')
    for i = 1:numel(L1s(:,idx_K))
        text(L1s(i),L2s(i),['$\alpha = ' num2str(alphas(i)) '$'],'Interpreter','latex')
    end
    xlabel('$L_1$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    hold off
end

% L, L1, L2
for idx_K=1:length(Ks)
    figure
    subplot(1,3,1)
    hold on
    plot(alphas,Ls(:, idx_K),'r*-')
    xlabel('$\alpha$','Interpreter','latex')
    ylabel('$L$','Interpreter','latex')
    subplot(1,3,2)
    hold on
    plot(alphas,L1s(:, idx_K),'b*-')
    xlabel('$\alpha$','Interpreter','latex')
    ylabel('$L_1$','Interpreter','latex')
    subplot(1,3,3)
    hold on
    plot(alphas,L2s(:, idx_K),'m*-')
    xlabel('$\alpha$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    hold off
end

end

end

fprintf("Adamar -> Mean learning F1-score: %.2f | Mean testing F1-score: %.2f\n", mean(lf1score), mean(tf1score));
if SVM
    fprintf("SVM ----> Mean learning F1-score: %.2f | Mean SVM testing F1-score: %.2f\n",mean([SVM_stats_train.f1score]), mean([SVM_stats_test.f1score]));
end
