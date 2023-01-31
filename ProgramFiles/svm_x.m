function [stats_train, stats_test] = svm_x(X, testing_images, descriptors, ca_Y, dataset)
%SVM_X SVM with raw data
%   X ... matrix of descriptors [n_rows , 1:(number_of_feature + 1) ]
arguments
    X {mustBeNumeric}
    testing_images {mustBeNumeric} = 68
    descriptors = [ Descriptor.Roughness, Descriptor.Color ]
    ca_Y = []
    dataset = 'Dataset'
end

%SVMModel = fitcsvm(X(:,1:4), X(:,5),'Standardize',false,'KernelFunction','polynomial','PolynomialOrder',2,'KernelScale','auto');
SVMModel = fitcsvm(X(:,1:end-1), X(:,end),'Standardize',false,'KernelFunction','gaussian','KernelScale','auto', 'Verbose',1);
%[SVMModel, FitInfo] = fitclinear(X(:,1:4), X(:,5), 'GradientTolerance', 1e-8); % Unsemicolon for results
% fprintf("Performing SVM cross-validation...\n");
% CVSVMModel = crossval(SVMModel);
% classLoss = kfoldLoss(CVSVMModel);
% fprintf("ClassLoss = %f\n", classLoss);
[labels_train,~] = predict(SVMModel,X(:,1:end-1));
stats_train = statistics(labels_train, X(:,end));

if isempty(ca_Y)
    %Load testing data
    ca_pred = load_images(testing_images);
    Y = get_descriptors(ca_pred, descriptors);
    % Y(:, 1) = (Y(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)
    
    %Predict
    [labels_test,~] = predict(SVMModel,Y(:,1:end-1));
    stats_test = statistics(labels_test, Y(:,end));
    visualize(ca_pred(1,1), ca_pred(1,2), Y(:,end), labels_test, "SVM X");
else
    n = numel(ca_Y);
    for i = 1:n
        Y = ca_Y{i}.X;
        
        [labels_test,~] = predict(SVMModel, Y(:,1:end-1));
        stats_test = statistics(labels_test, Y(:,end));
        
        if i == 1
            stats_avg = stats_test;
        else
            stats_avg.precision = stats_test.precision + stats_avg.precision;
            stats_avg.recall = stats_test.recall + stats_avg.recall;
            stats_avg.f1score = stats_test.f1score + stats_avg.f1score;
            stats_avg.accuracy = stats_test.accuracy + stats_avg.accuracy;
        end
    
        if strcmp(dataset, 'Dataset')
            original_rgb{1} = imread(sprintf('Dataset/Original/%d.jpg', ca_Y{i}.I));
            annnotation{1} = imread(sprintf('Dataset/Annotations/%d.png', ca_Y{i}.I));
        else
            original_rgb{1} = imread(sprintf('Dataset2/Original/%d.jpeg', ca_Y{i}.I));
            annnotation{1} = imread(sprintf('Dataset2/Annotations/%d.png', ca_Y{i}.I));
        end
        
        visualize(original_rgb, annnotation, Y(:,end), labels_test, sprintf('SVM X'));
        pause(1)
    end
    stats_avg.precision = stats_avg.precision / n;
    stats_avg.recall = stats_avg.recall / n;
    stats_avg.f1score = stats_avg.f1score / n;
    stats_avg.accuracy = stats_avg.accuracy / n;
    stats_test = stats_avg;
end


end

