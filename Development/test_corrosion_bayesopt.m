%close all
clear all
addpath(genpath(pwd));
rng(13);
MRMR = [];
PCA = [];
SCALE = [];
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
MCC = 0.676 (i.e. 0.838) % slightly better test perf.

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
%descriptors = ["StatMomHSV", "GLCM_Gray"];
%descriptors = ["StatMomHSV","GLCM_HSV","LBP_HSV"];
descriptors = ["StatMomHSV","StatMomRGB","GLCM_Gray","GLCM_HSV","GLCM_RGB"];
descriptorsEnum = map_descriptors(descriptors);

gt = 'GroundTruthBinaryCropped';
undersample = false;
[X, ca_Y] = get_data_from_dataset_selection(0.7, gt, descriptors, undersample);
X(isnan(X)) = 0; 
y = X(:,end);
[X, ca_Y, ~, SCALE] = scaling(X(:,1:end-1), ca_Y, 'minmax');
%[X, ca_Y, ~, ~] = scaling(X(:,1:end-1), ca_Y, 'zscore');
%[X, ca_Y, ~, ~] = scaling(X(:,1:end-1), ca_Y, 'zscore', 'robust');

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

[X_train, X_test, y_train, y_test] = train_test_split(X,y,0.7);

%kmeans_criteria(X_train);
test_standard_classifiers(X_train,X_test,y_train,y_test);

% Train KKLDJ
mdl = KKLDJ();
mdl.MRMR = MRMR;
mdl.PCA = PCA;
mdl.SCALE = SCALE;
mdl.NCAidx = NCAidx;
mdl.Nrand = 5;
mdl.scaleT = false;
mdl.descriptors = descriptorsEnum;

alphas = optimizableVariable('alpha',[0.5,0.9999],'Type','real');
clusters = optimizableVariable('K',[2,100],'Type','integer');

fun = @(x)myobjectivefun(x,X,y,mdl);

% 'PlotFcn', 'all',... %'PlotFcn', @plotMinObjective,...
results = bayesopt(fun,[alphas,clusters],'IsObjectiveDeterministic',false,...
    'PlotFcn', {@plotMyObjectiveFun,@plotObjectiveModel,@plotMinObjective,@plotElapsedTime},...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'Verbose',1,'UseParallel',true,'MaxObjectiveEvaluations',30,...
    'NumCoupledConstraints',1);

best = bestPoint(results);
best_alpha = best.alpha;
best_K = best.K;

%--------------------------------------------------------------------------
% Evaluation on the test set
%--------------------------------------------------------------------------

if false
    % Test
    mdl.alpha = best_alpha;
    mdl.K = best_K;
    mdl.fit(X_train,y_train);
    y_pred = mdl.predict(X_test');
    stats_test = statistics(y_pred, y_test');
    plot_corrosion_confmat2(y_test', y_pred);
    fprintf("\nTest performance:\nF1-score = %.3f\nAccuracy = %.3f\nnMCC = %.3f\n",...
        stats_test.f1score, stats_test.accuracy, stats_test.nmcc);
end

% test K=86, a=0.9728

fprintf("Fitting optimal hyperparameters... (this may take a while)\n")
n_rand = 3;
stats_test = cell(n_rand,1);
models = cell(n_rand,1);

% Parallel training of #n_rand models
parfor n = 1:n_rand
    % Shuffle the data:
    [X_train, X_test, y_train, y_test] = train_test_split(X,y,0.7);

    models{n} = KKLDJ();
    models{n}.PCA = PCA;
    models{n}.SCALE = SCALE;
    models{n}.Nrand = 5;
    models{n}.scaleT = false;
    models{n}.descriptors = descriptorsEnum;
    models{n}.alpha = best_alpha;
    models{n}.K = best_K;
    models{n}.fit(X_train,y_train);
    y_pred = models{n}.predict(X_test');
    stats_test{n} = statistics(y_pred, y_test');
    %plot_corrosion_confmat(stats_test{n})
    %plot_corrosion_confmat2(y_test', y_pred)
    %fprintf("\nTest performance:\nF1-score = %.3f\nAccuracy = %.3f\nnMCC = %.3f\n",...
    %         stats_test{n}.f1score, stats_test{n}.accuracy, stats_test{n}.nmcc);
end
stats_test_vec = [stats_test{:}];
nmcc = mean([stats_test_vec.nmcc]);
acc  = mean([stats_test_vec.accuracy]);
f1s  = mean([stats_test_vec.f1score]);
std_nmcc = std([stats_test_vec.nmcc]);
std_acc  = std([stats_test_vec.accuracy]);
std_f1s  = std([stats_test_vec.f1score]);

fprintf("- Average test performance: ");
fprintf("nMCC = %.3f ± %.2f  |  F1 = %.3f ± %.2f  |  ACC = %.3f ± %.2f  |\n",...
        nmcc,std_nmcc,f1s,std_f1s,acc,std_acc);

% Mean confusion matrix
conf_mats = reshape([stats_test_vec(:).CM],2,2,[]);
cm = round(mean(conf_mats, 3));
std_cm = round(std(conf_mats(1,1,:),0,3));

% Select the best model
[best_nmcc, idx] = max([stats_test_vec.nmcc]);
mdlopt = models{idx};
y_pred = mdlopt.predict(X_test');
stats_final = statistics(y_pred, y_test');
plot_corrosion_confmat(stats_final);
plot_corrosion_confmat2(y_test',y_pred);

fprintf("\n- Final test performance:\n");
fprintf("  nMCC = %.3f |  F1 = %.3f  |  ACC = %.3f  |\n",...
        stats_final.nmcc,stats_final.f1score,stats_final.accuracy);

%--------------------------------------------------------------------------
% ROC curve, P-R curve
%--------------------------------------------------------------------------

% Plot Precision-Recall curve and calculte AUC-Precision-Recall
[AUCpr] = plot_prec_rec(y_test',y_pred);

% Plot ROC curve and calculte AUC
[TPR,FPR,T,AUC] = plot_roc(y_test',y_pred);

% Find the threshold that maximizes the balance between
% sensitivity and specificity (Youden's J statistic)
J = max(abs(TPR-FPR));
threshold = T(find(abs(TPR-FPR)==J,1));
mdlopt.threshold = threshold;
% Classify new data using threshold
y_pred_J = mdlopt.predict(X_test') > threshold;
stats_test_J = statistics(double(y_pred_J), y_test');
fprintf("\nTest performance (Youden's J statistic):\nF1-score = %.3f\nAccuracy = %.3f\nnMCC = %.3f\n\n",...
    stats_test_J.f1score, stats_test_J.accuracy, stats_test_J.nmcc);



%--------------------------------------------------------------------------
% Visualization (corrosion overlay)
%--------------------------------------------------------------------------

for i = 1:length(ca_Y)
    f{i} = mdlopt.view(ca_Y{i});
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

%--------------------------------------------------------------------------
% Save the pre-trained model to a ".mat" file (optional)
%--------------------------------------------------------------------------

if false
    mdlopt.description = "close to perfect Lambda (99% binary);"+...
        "bayesopt: a=[0.01,0.9999], K=[2,100]; K=81; alpha=0.8827;"+...
        "PCA; SM(all), GLCM(all), no undersampling";
    save('Model_v2.mat', 'mdlopt', '-v7.3'); 
    save('Model_v2_test.mat', 'ca_Y', '-v7.3'); 
end
%Load the pre-trained model 
% my_mdl = load('ModelKKLDJ.mat').mdl;
% my_mdl = load('ModelKKLDJ_bayesopt_04-24.mat').mdlopt;


%--------------------------------------------------------------------------
% Bayesian hyperparameter optimization
%--------------------------------------------------------------------------

function [objective, constraint, Lfvals] = ...
    myobjectivefun(x, X, y, mdl)
%MYOBJECTIVE

    [X_train, X_test, y_train, y_test] = train_test_split(X,y,0.7);
    mdl.alpha = x.alpha;
    mdl.K = x.K;
    mdl.fit(X_train,y_train);
    
    Lfvals = mdl.L;
    
    % If the of L2 is less than 50,
    % the constraint is positive,
    % thus unsatisfied.
    constraint = - mdl.L.L2 + 50; 
    
    %constraint = [];
    
    y_pred = mdl.predict(X_test');
    stats = statistics(y_pred, y_test');
    %objective = - stats.mcc;
    objective = - stats.prauc;
end

function stop = plotMyObjectiveFun(results,state)
persistent hs LsTrace
stop = false;
switch state
    case 'initial'
        hs = figure;
    case 'iteration'
        figure(hs)
        
        % get last fval from UserDataTrace property
        lastL = results.UserDataTrace{end}; 
        
        % accumulate function vals in a vector
        LsTrace{end+1} = lastL;
        LsTraceVec = [LsTrace{:}];
        
        plot(1:length(LsTrace),[LsTraceVec.L],'r*-', 'LineWidth', 2)
        hold on
        plot(1:length(LsTrace),[LsTraceVec.L1],'b*-.', 'LineWidth', 2)
        plot(1:length(LsTrace),[LsTraceVec.L2],'c*-.', 'LineWidth', 2)
        line(xlim, [100 100], 'Color', 'k', 'LineWidth', 2);
        set(gca, 'yscale', 'log');
        set(gca, 'FontSize', 14);
        xlabel ('Iteration number', 'Interpreter', 'latex')
        ylabel ('$L$' , 'Interpreter', 'latex')
        legend('$L$','$L_1$','$L_2$', 'Interpreter', 'latex')
        drawnow
end
end
