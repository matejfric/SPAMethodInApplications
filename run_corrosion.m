%--------------------------------------------------------------------------
% Corrosion detection with K-means+KLD+Jensen (KKLDJ)
%--------------------------------------------------------------------------

clear all
addpath(genpath(pwd));
rng(42);

%{

This script was used to train the model presented in Section 5.7. Note that
the results may not be the same due to the stochastic nature of Bayesian
optimization. To replicate the results in Section 5.7 run the script
'run_corrosion_mwe.m'.

Expected runtime with test size 0.7 is 40 minutes with 6 core CPU
(with parallelization). Therefore, the test size is set to 0.1 (feel free
to change it).

Updates will be commited to "Development/test_corrosion_bayesopt.m".

%}

% The amount of training data is set to 10%! 
% For the final model this parameter was set to 0.7. 
train_size = 0.1;
gt = 'GroundTruth';
descriptors = ["StatMomHSV","StatMomRGB","GLCM_Gray","GLCM_HSV","GLCM_RGB"];
undersample = false;
[X, ca_Y] = get_data_from_dataset_selection(0.1, gt, descriptors, undersample);
X(isnan(X)) = 0; 
y = X(:,end);
[X, ca_Y, ~, SCALE] = scaling(X(:,1:end-1), ca_Y, 'minmax');

% Principal Component Analysis (PCA)
[X, ~, PCA] = mypca(X, [], y);

[X_train, X_val, X_test, y_train, y_val, y_test] = train_test_val_split(X,y,0.6,0.2);

%test_standard_classifiers(X_train,X_test,y_train,y_test);

%--------------------------------------------------------------------------
% Training / hyperparameter tuning
%--------------------------------------------------------------------------

mdl = KKLDJ();
mdl.PCA = PCA;
mdl.SCALE = SCALE;
mdl.Nrand = 5;
mdl.scaleT = false;
mdl.descriptors = map_descriptors(descriptors);

alphas = optimizableVariable('alpha',[0.99,1-1e-8],'Type','real'); % [0.99,0.99999999]
clusters = optimizableVariable('K',[2,150],'Type','integer');

fun = @(x)myobjectivefun(x,X_train, y_train, X_val, y_val, mdl);

% 'PlotFcn', 'all',... %'PlotFcn', @plotMinObjective,...
results = bayesopt(fun,[alphas,clusters],'IsObjectiveDeterministic',false,...
    'PlotFcn', {@plotMyObjectiveFun,@plotObjectiveModel,@plotMinObjective,@plotElapsedTime},...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'Verbose',1,'UseParallel',true);
    %'NumCoupledConstraints',1);

best = bestPoint(results);
best_alpha = best.alpha;
best_K = best.K;

%--------------------------------------------------------------------------
% Evaluation on the test set
%--------------------------------------------------------------------------

if false
    mdl.alpha = best_alpha;
    mdl.K = best_K;
    mdl.fit(X_train,y_train);
    y_pred = mdl.predict(X_test');
    stats_test = statistics(y_pred, y_test');
    plot_corrosion_confmat2(y_test', y_pred);
    fprintf("\nTest performance:\nF1-score = %.3f\nAccuracy = %.3f\nnMCC = %.3f\n",...
        stats_test.f1score, stats_test.accuracy, stats_test.nmcc);
end

fprintf("Fitting optimal hyperparameters... (this may take a while)\n")
n_rand = 5;
stats_test = cell(n_rand,1);
models = cell(n_rand,1);

% Parallel training of #n_rand models
parfor n = 1:n_rand
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
    plot_corrosion_confmat(stats_test{n})
    plot_corrosion_confmat2(y_test', y_pred)
    fprintf("\nModel %d test performance:\nF1-score = %.3f\nAccuracy = %.3f\nnMCC = %.3f\n",...
        n,stats_test{n}.f1score,stats_test{n}.accuracy,stats_test{n}.nmcc);
end
stats_test_vec = [stats_test{:}];
nmcc = mean([stats_test_vec.nmcc]);
acc  = mean([stats_test_vec.accuracy]);
f1s  = mean([stats_test_vec.f1score]);
std_nmcc = std([stats_test_vec.nmcc]);
std_acc  = std([stats_test_vec.accuracy]);
std_f1s  = std([stats_test_vec.f1score]);

fprintf("\nAverage test performance:\n");
fprintf("nMCC = %.3f ± %.2f  |  F1 = %.3f ± %.2f  |  ACC = %.3f ± %.2f\n",...
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

fprintf("\nFinal test performance:\n");
fprintf("nMCC = %.3f |  F1 = %.3f  |  ACC = %.3f\n",...
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
    mdlopt.description = "no undersampling; bayesopt; K=136; alpha=0.9901;"+...
        "PCA; SM(all), GLCM(all)";
    save('MyModelKKLDJ.mat', 'mdlopt', '-v7.3'); 
end
%Load the pre-trained model 
% my_mdl = load('MyModelKKLDJ.mat').mdl;

%--------------------------------------------------------------------------
% Bayesian hyperparameter optimization
%--------------------------------------------------------------------------

% Objective function for Bayesian optimization
function [objective, constraint, Lfvals] = ...
    myobjectivefun(x, X_train, y_train, X_val, y_val, mdl)

    mdl.alpha = x.alpha;
    mdl.K = x.K;
    mdl.fit(X_train,y_train);
    
    Lfvals = mdl.L;
    
    constraint = [];
    
    y_pred = mdl.predict(X_val');
    stats_val = statistics(y_pred, y_val');
    objective = - stats_val.mcc;
end

% Custom utility function for plotting objective function value (L,L1,L2)
% during Bayesian optimization.
function stop = plotMyObjectiveFun(results,state)
persistent hs LsTrace
stop = false;
switch state
    case 'initial'
        hs = figure;
    case 'iteration'
        figure(hs)
        
        % Get last fval from UserDataTrace property
        lastL = results.UserDataTrace{end}; 
        
        % Accumulate function vals in a vector
        LsTrace{end+1} = lastL;
        LsTraceVec = [LsTrace{:}];
        
        plot(1:length(LsTrace),[LsTraceVec.L],'r-', 'LineWidth', 2)
        hold on
        plot(1:length(LsTrace),[LsTraceVec.L1],'b-.', 'LineWidth', 2)
        plot(1:length(LsTrace),[LsTraceVec.L2],'c-.', 'LineWidth', 2)
        set(gca, 'FontSize', 14);
        xlabel ('Iteration number', 'Interpreter', 'latex')
        ylabel ('L' , 'Interpreter', 'latex')
        legend('L','L1','L2', 'Interpreter', 'latex')
        drawnow
end
end

