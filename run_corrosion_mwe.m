%--------------------------------------------------------------------------
% Corrosion Detection - Minimal Working Example (MWE)
%--------------------------------------------------------------------------

clear all
addpath(genpath(pwd));
rng(42); % For reproducibility; will guarantee that the model is not tested
         % on training data and the results will correspond to Fig. 5.3 and
         % Fig. 5.4 in the main text (page 50).

% Load pre-trained model (see class KKLDJ for more information)
mdlopt = load('PretrainedModel.mat').mdlopt; 

% Load PCA parameters
PCA = mdlopt.PCA; 

% Load the data
[X,images] = get_data_from_dataset_selection();
y = X(:,end);

% Apply normalization and PCA to data
[X,images] = scaling(X(:,1:end-1),images,'minmax');
X = (X-PCA.mu)*PCA.coeff(:,1:PCA.idx);

% Perform three-way-split
[X_train,~,X_test,y_train,~,y_test] = train_test_val_split(X,y,0.6,0.2);

% Make a predicition based on Bayesian inference 
y_pred = mdlopt.predict(X_test');

% Compute performance metrics
stats = statistics(y_pred,y_test');

% Plot confusion matrix
plot_corrosion_confmat2(y_test',y_pred);

%--------------------------------------------------------------------------
% Visualization (corrosion overlay)
%--------------------------------------------------------------------------

% Cell array images contains features extracted from withheld images 
% (see Section 5.7 for more information).

fprintf("\nPress a button to continue... (press 'q' to quit)\n");
for i = 1:length(images)
    fig = view(mdlopt, images{i});
    pause(0.1)
    % Wait for a press
    try
    if waitforbuttonpress == 1 
        % Check which key was pressed
        key = get(gcf,'CurrentCharacter');
        % Press 'q' to break out of the loop
        if strcmp(key,'q') 
            % Close all images
            close([f{:}])
            break
        end
    end
    catch ME
        close all
        break
    end
end

%--------------------------------------------------------------------------
% Current limitations
%--------------------------------------------------------------------------
%{

- Further analysis of the optimal image features is necessary.
- Slow training because of the non-trivial tuning of hyperparameters.

%}

function fig = view(obj, img, dataset)
    arguments
        obj, img, dataset = "DatasetSelection"
    end
    X = img.X(:,1:end-1);
    if ~isempty(obj.SCALE)
        X = scaling(X, [], 'minmax', [], obj.SCALE.colmin, obj.SCALE.colmax);
    else
        X = scaling(X, [], 'zscore');
    end
    if ~isempty(obj.PCA)
        % Apply PCA
        X = (X-obj.PCA.mu)*obj.PCA.coeff(:,1:obj.PCA.idx);
    end
    idx = img.I; 
    ground_truth = round(img.X(:,end));
    prediction = predict_bayes(obj.mdl, X')';
    stats_test = statistics(prediction, ground_truth);
    fprintf("\nImage %d:\nF1-score = %.3f\nAccuracy = %.3f\nNormMCC = %.3f\n\n",...
        idx, stats_test.f1score, stats_test.accuracy, stats_test.mcc);
    [original_rgb, annnotation] = load_img_annot(dataset,idx);
    fig = visualize(original_rgb, annnotation, ground_truth, prediction);

end