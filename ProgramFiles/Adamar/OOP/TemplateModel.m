classdef TemplateModel < handle
    %TemplateModel - Abstract class machine learning model 
    
    properties
        mdl         = [] % Model struct
        statsTrain  = [] 
        L           = []
        descriptors = []
        SCALE       = []       % Min-Max scaling parameters
        MRMR        = []       % Minimum Redundancy Maximum Relevance algorithm
        PCA         = []       % Principal Component Analysis
        K           = 25       % Number of clusters
        alpha       = 0.5      % Regularization parameter
        maxIt       = 100      % Maximum number of iterations
        Nrand       = 5        % Number of random runs
        scaleT      = true     % Scaling of L1
        verbose     = false
    end
    
    methods (Abstract)
        fit(obj, X_train, y_train)
    end
    
    methods
        function obj = TemplateModel(K,alpha,maxIt,Nrand,scaleT)
            % Constructor
            arguments
                K      = 25       % Number of clusters
                alpha  = 0.5;     % Regularization parameter
                maxIt  = 100;     % Maximum number of iterations
                Nrand  = 5;       % Number of random runs
                scaleT = true;    % Scaling of L1
            end
            obj.K = K;
            obj.alpha = alpha;
            obj.maxIt = maxIt;
            obj.Nrand = Nrand;
            obj.scaleT = scaleT;
        end
        
        function [y_pred, labels_pred] = predict(obj, X_new)
            % Use the trained model to predict the class labels of new data
            [y_pred, labels_pred] = predict_bayes(obj.mdl, X_new);
        end
        
        function stats = computeStats(obj, y_pred, y_test)
            if numel(obj.mdl.classes) > 2
                stats = statistics_multiclass(y_pred, y_test);
            else
                stats = statistics(y_pred, y_test);
            end
        end
        
        function view(obj, ca_new, dataset)
            arguments
                obj, ca_new, dataset = "DatasetSelection"
            end
            X = ca_new.X(:,1:end-1);
            X = scaling(X, [], 'minmax', [], obj.SCALE.colmin, obj.SCALE.colmax);
            if ~isempty(obj.MRMR)
                % Apply MRMR
                X = X(:, obj.MRMR.scores > obj.MRMR.limit);
            end
            if ~isempty(obj.PCA)
                % Apply PCA
                X = (X-obj.PCA.mu)*obj.PCA.coeff(:,1:obj.PCA.idx);
            end
            idx = ca_new.I; 
            ground_truth = round(ca_new.X(:,end));
            prediction = predict_bayes(obj.mdl, X')';
            stats_test = statistics(prediction, ground_truth);
            fprintf("\nImage %d:\nF1-score = %.3f\nAccuracy = %.3f\n\n",...
                idx, stats_test.f1score, stats_test.accuracy);
            [original_rgb, annnotation] = load_img_annot(dataset,idx);
            visualize(original_rgb, annnotation, ground_truth, prediction)
        end
        
        function processedImg = processImage(obj, img)
            arguments
                obj, img = imread('confusion_matrix.png')
            end
            ca_img{1} = img;
            X = get_descriptors(ca_img, obj.descriptors);
            if isempty(obj.SCALE.colmin) || isempty(obj.SCALE.colmax)
                X = scaling(X, [], 'zscore', 'robust');
            else
                X = scaling(X, [], 'minmax', [], obj.SCALE.colmin, obj.SCALE.colmax);
            end
            if ~isempty(obj.MRMR)
                % Apply MRMR
                X = X(:, obj.MRMR.scores > obj.MRMR.limit);
            end
            if ~isempty(obj.PCA)
                % Apply PCA
                X = (X-obj.PCA.mu)*obj.PCA.coeff(:,1:obj.PCA.idx);
            end
            prediction = predict_bayes(obj.mdl, X')';
            processedImg = corrosion_overlay(ca_img,prediction);
        end
        
        function printStatsTrain(obj)
            mf1 = mean([obj.statsTrain.f1score]);
            macc = mean([obj.statsTrain.accuracy]);
            fprintf("\nTraining results:\nMean Accuracy = %.3f\nMean F1-score = %.3f\n",...
            macc, mf1);
        end
    end
end

