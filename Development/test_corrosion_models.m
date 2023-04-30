%close all
clear all
addpath(genpath(pwd));
rng(42);
MRMR = [];
PCA = [];
NCAidx = [];

%{

ScaleT = true;

Training results:
Mean F1-score = 0.829
Mean Accuracy = 0.835

Test performance:
F1-score = 0.844
Accuracy = 0.834
MCC = 0.673

---------------------------------------------------------------------------

ScaleT = false;

Training results:
Mean F1-score = 0.826
Mean Accuracy = 0.833

Test performance:
F1-score = 0.843
Accuracy = 0.837
MCC = 0.676 % slightly better test perf.

---------------------------------------------------------------------------

no undersampling (F1-score will be different and to some extent biased!):
- 40 minutes training
- Ks = [25,35,45,55,65,75];
- alphas = 0.9999:0.00002:1;

Training results:
Mean F1-score = 0.822
Mean Accuracy = 0.775

Test performance:
F1-score = 0.798
Accuracy = 0.835
nMCC = 0.830

%}

% [X,y] = get_parking_data()

%descriptors = ["LBP" "LBP_HSV" "LBP_RGB" "StatMomHSV" "StatMomRGB" "GLCM_HSV" "GLCM_RGB", "GLRLM"];
%descriptors = ["LBP" "LBP_HSV" "LBP_RGB"]; % 0.6, 0.8, 0.7
%descriptors = ["GLCM_Gray"]; %78
%descriptors = ["GLCM_HSV"]; %83 Informative feature
%descriptors = ["StatMomHSV"];
%descriptors = ["StatMomHSV","GLCM_HSV","LBP_HSV"];
descriptors = ["StatMomHSV","StatMomRGB","GLCM_Gray","GLCM_HSV","GLCM_RGB"];
descriptorsEnum = map_descriptors(descriptors);

gt = 'GroundTruthBinaryCropped';
undersample = false;
[X, ca_Y] = get_data_from_dataset_selection(0.8, gt, descriptors, undersample);
X(isnan(X)) = 0; 
y = X(:,end);
[X, ca_Y, ~, SCALE] = scaling(X(:,1:end-1), ca_Y, 'minmax');
%[X, ca_Y, ~, SCALE] = scaling(X(:,1:end-1), ca_Y, 'zscore', 'robust');

%Neighborhood Component Analysis (NCA)
% [NCAidx] = mynca(X,y)
% [tbl] = descriptors2table(X,y);
% sel_tbl = tbl(:,NCAidx);
%Output from (trained) NCA:
% NCAidx = [14,35,41,44,49,50,61,62,67,74,91,107,116,123,132,133,157,171,173]; 
% X = X(:,NCAidx);

%Rank features for classification using minimum redundancy maximum relevance (MRMR) algorithm
% [idx,mrmr_scores] = fscmrmr(X,y);
% figure
% bar(mrmr_scores(idx))
% xlabel('Predictor rank')
% ylabel('Predictor importance score')
% MRMR.scores = mrmr_scores;
% MRMR.limit = 0.15;
% X = X(:, mrmr_scores > MRMR.limit);

%Univariate feature ranking for classification using chi-square tests
% [idx,scores] = fscchi2(X,y);
% figure
% bar(scores(idx))
% xlabel('Predictor rank')
% ylabel('Predictor importance score')

%Principal Component Analysis (PCA)
[X, ~, PCA] = mypca(X, [], y);

[X_train, X_val, X_test, y_train, y_val, y_test] = train_test_val_split(X,y,0.6,0.2);

%kmeans_criteria(X_train);

%test_standard_classifiers(X_train,X_test,y_train,y_test);

% Train KKLDJ
mdl = KKLDJ();
mdl.MRMR = MRMR;
mdl.PCA = PCA;
mdl.SCALE = SCALE;
mdl.NCAidx = NCAidx;
mdl.Nrand = 5;
mdl.scaleT = false;
mdl.descriptors = descriptorsEnum;
alphas = 0:0.1:1;
%alphas = [0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999];%, 0.99995, 0.99999];
%alphas = 0.99:0.002:1;
%alphas = 0.999:0.0002:1;
%alphas = 0.99975:0.00005:1;
% Ks = [5,10,15];
Ks = [10,20,30,40];
%Ks = [30,50,80];
%Ks = [55,65,75,85,95];
%Ks = [55,65,75];
%Ks = [43,45,47,49];
%Ks = 100;

Ls  = zeros(numel(alphas),length(Ks));
L1s = zeros(numel(alphas),length(Ks));
L2s = zeros(numel(alphas),length(Ks));

% Training/validation
fprintf("Training...\n");
for a = progress(1:length(alphas))
    for k = 1:length(Ks)
        mdl.alpha = alphas(a);
        mdl.K = Ks(k);
        mdl.fit(X_train,y_train);
        
        Ls(a,k)  = mdl.L.L;
        L1s(a,k) = mdl.L.L1;
        L2s(a,k) = mdl.L.L2;
        
        y_pred = mdl.predict(X_val');
        stats_val(a,k)  = statistics(y_pred, y_val');
        stats_train(a,k) = mdl.statsTrain;
    end
end
mdl.statsTrain = stats_train;
mdl.printStatsTrain();

if true
    for k = 1:length(Ks)
        plot_L_curves(Ls(:,k), L1s(:,k), L2s(:,k), alphas, Ks(k));
        plot_f1score(stats_train(:,k), stats_val(:,k), alphas, Ks(k));
    end
end

% Grid search
%[best_alpha, best_K, best_f1s] = grid_search(stats_val,alphas,Ks);
[best_alpha, best_K, best_mcc] = grid_search_mcc(stats_val,alphas,Ks);

% Test
fprintf("Fitting optimal hyperparameters...\n")
n_rand = 1;
stats_test = cell(n_rand,1);
models = cell(n_rand,1);
for n = progress(1:n_rand)
    models{n} = KKLDJ();
    models{n}.MRMR = MRMR;
    models{n}.PCA = PCA;
    models{n}.SCALE = SCALE;
    models{n}.NCAidx = NCAidx;
    models{n}.Nrand = 5;
    models{n}.scaleT = false;
    models{n}.descriptors = descriptorsEnum;
    
    
    models{n}.alpha = best_alpha;
    models{n}.K = best_K;
    models{n}.fit(X_train,y_train);
    y_pred = models{n}.predict(X_test');
    stats_test{n} = statistics(y_pred, y_test');
    plot_corrosion_confmat(stats_test{n})
    plot_corrosion_confmat2(y_test', y_pred)
    fprintf("\nTest performance:\nF1-score = %.3f\nAccuracy = %.3f\nnMCC = %.3f\n",...
        stats_test{n}.f1score, stats_test{n}.accuracy, stats_test{n}.nmcc);
end
stats_tmp = [stats_test{:}];
final_nmcc = mean([stats_tmp.nmcc]);

% Plot Precision-Recall curve and calculte AUC-Precision-Recall
[AUCpr] = plot_prec_rec(y_test',y_pred);

% Plot ROC curve and calculte AUC
[TPR,FPR,T,AUC] = plot_roc(y_test',y_pred);

% Find the threshold that maximizes the balance between
% sensitivity and specificity (Youden's J statistic)
J = max(abs(TPR-FPR));
threshold = T(find(abs(TPR-FPR)==J,1));
% Classify new data using threshold
y_pred_J = mdl.predict(X_test') > threshold;
stats_test_J = statistics(double(y_pred_J), y_test');
fprintf("\nTest performance (Youden's J statistic):\nF1-score = %.3f\nAccuracy = %.3f\nnMCC = %.3f\n\n",...
    stats_test_J.f1score, stats_test_J.accuracy, stats_test_J.nmcc);

% Visualization (corrosion overlay)
for i = 1:length(ca_Y)
    f{i} = mdl.view(ca_Y{i});
    pause(0.1)
    % Wait for a press
    if waitforbuttonpress == 1 
        % Check which key was pressed
        key = get(gcf,'CurrentCharacter');
        % Break out of loop if 'q' is pressed
        if strcmp(key,'q') 
            % Close all images
            close([f{:}])
            break;
        end
    end
end




%Save the pre-trained model (object of class KKLDJ) to a .mat file
% mdl.description = "no undersampling, K=95, alpha=0.9998, PCA";
% save('ModelKKLDJ_ALL_PCA_04-23.mat', 'mdl', '-v7.3'); 
%Load the pre-trained model 
% my_mdl = load('ModelKKLDJ.mat').mdl;


