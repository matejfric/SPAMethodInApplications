function [stats_train, stats_test] = svm_kmeans(X, K, testing_images, descriptors, ca_Y)
%KMEANS_SVM Summary of this function goes here
%   X ... matrix of descriptors [n_rows , 1:(number_of_feature + 1) ]
arguments
    X {mustBeNumeric}
    K (1,1) {mustBeNumeric} = 10
    testing_images {mustBeNumeric} = 68
    descriptors = []
    ca_Y = [] % Testing matrices
end

% Remember scaling for testing dataset
% a = min(X(:, 1));
% b = max(X(:, 1));
% X(:, 1) = (X(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

[idx, C, ~] = kmeans(X(:,1:end-1),K, 'MaxIter',1000);
cluster_labels = (1:K);
PiX = onehotencode(idx,2,"ClassNames", cluster_labels);

%[SVMModel_PiX] = fitcsvm(PiX, X(:,5),'Standardize',false,'KernelFunction','polynomial','KernelScale','auto');
[SVMModel_PiX, FitInfo_PiX] = fitclinear(PiX, X(:,end), 'GradientTolerance', 1e-8); % Unsemicolon for results
[labels_train,~] = predict(SVMModel_PiX, PiX);
stats_train = statistics(labels_train, X(:,end));

% fprintf("Performing SVM_PiX cross-validation...\n");
% CVSVMModel_PiX = crossval(SVMModel_PiX);
% classLoss_PiX = kfoldLoss(CVSVMModel_PiX);
% fprintf("ClassLoss = %f\n", classLoss_PiX);
% fprintf("\n");

if isempty(ca_Y)

    %Load testing data
    ca_pred = load_images(testing_images);
    Y = get_descriptors(ca_pred, descriptors);
    % Y = roughness_analysis(ca_predict);
    % Y(:, 1) = (Y(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

    %K-means for testing data
    dist_from_C = zeros(K, size(Y,1));
    for k=1:K
        dist_from_C(k,:) = sum((Y(:,1:end-1)' - C(k,:)').^2,1);
    end
    [~,idxY] = min(dist_from_C);
    GammaY = zeros(K,length(idxY));
    for k = 1:K
       GammaY(k,idxY==k) = 1; 
    end
    [labels_test,~] = predict(SVMModel_PiX, GammaY');
    stats_test = statistics(labels_test, Y(:,end));
    visualize(ca_pred(1,1), ca_pred(1,2), Y(:,end), labels_test, ["SVM K-means K=", num2str(K)]);
    %pause
else
    n = numel(ca_Y);
    for i = 1:n
        Y = ca_Y{1}.X;
        
        %K-means for testing data
        dist_from_C = zeros(K, size(Y,1));
        for k=1:K
            dist_from_C(k,:) = sum((Y(:,1:end-1)' - C(k,:)').^2,1);
        end
        [~,idxY] = min(dist_from_C);
        GammaY = zeros(K,length(idxY));
        for k = 1:K
           GammaY(k,idxY==k) = 1; 
        end
        
        [labels_test,~] = predict(SVMModel_PiX, GammaY');
        stats_test = statistics(labels_test, Y(:,end));
        
        if i == 1
            stats_avg = stats_test;
        else
            stats_avg.precision = stats_test.precision + stats_avg.precision;
            stats_avg.recall = stats_test.recall + stats_avg.recall;
            stats_avg.f1score = stats_test.f1score + stats_avg.f1score;
            stats_avg.accuracy = stats_test.accuracy + stats_avg.accuracy;
        end
    
        original_rgb{1} = imread(sprintf('Dataset2/Original/%d.jpeg', ca_Y{i}.I));
        annnotation{1} = imread(sprintf('Dataset2/Annotations/%d.png', ca_Y{i}.I));
        %     original_rgb{1} = imread(sprintf('Dataset/Original/%d.jpg', ca_Y{i}.I));
        %     annnotation{1} = imread(sprintf('Dataset/Annotations/%d.png', ca_Y{i}.I));
        visualize(original_rgb, annnotation, Y(:,end), labels_test, sprintf('SVM K-means, K=%d', K));
        pause(1)
    end
    stats_avg.precision = stats_avg.precision / n;
    stats_avg.recall = stats_avg.recall / n;
    stats_avg.f1score = stats_avg.f1score / n;
    stats_avg.accuracy = stats_avg.accuracy / n;
    stats_test = stats_avg;
end

end

