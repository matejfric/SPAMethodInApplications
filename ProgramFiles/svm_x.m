function [stats_train, stats_test] = svm_x(X, testing_images, descriptors)
%SVM_X SVM with raw data
%   X ... matrix of descriptors [n_rows , 1:(number_of_feature + 1) ]
arguments
    X {mustBeNumeric}
    testing_images {mustBeNumeric} = 68
    descriptors = [ Descriptor.Roughness, Descriptor.Color ]
end

%SVMModel = fitcsvm(X(:,1:4), X(:,5),'Standardize',false,'KernelFunction','polynomial','PolynomialOrder',2,'KernelScale','auto');
SVMModel = fitcsvm(X(:,1:end-1), X(:,end),'Standardize',false,'KernelFunction','gaussian','KernelScale','auto');
%[SVMModel, FitInfo] = fitclinear(X(:,1:4), X(:,5), 'GradientTolerance', 1e-8); % Unsemicolon for results
% fprintf("Performing SVM cross-validation...\n");
% CVSVMModel = crossval(SVMModel);
% classLoss = kfoldLoss(CVSVMModel);
% fprintf("ClassLoss = %f\n", classLoss);

%Load testing data
ca_pred = load_images(testing_images);
Y = get_descriptors(ca_pred, descriptors);
% Y = roughness_analysis(ca_predict);
% Y(:, 1) = (Y(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

%Predict
[labels_train,~] = predict(SVMModel,X(:,1:end-1));
[labels_test,~] = predict(SVMModel,Y(:,1:end-1));

stats_train = statistics(labels_train, X(:,end));
stats_test = statistics(labels_test, Y(:,end));

visualize(ca_pred(1,1), ca_pred(1,2), Y(:,end), labels_test, "SVM X");

% err = norm(abs(label2 - Y(:,end)),1) / size(label1, 1);
% errKmeans = norm(abs(label1 - Y(:,end)),1) / size(label1, 1);
% fprintf("\nSVM with raw data error:         %.2f (Corroded patches predicted %.2f Vs. true %.2f),\nSVM with Kmeans and Gamma error: %.2f (Corroded patches predicted %.2f Vs. true %.2f)",...
%     err, sum(label1), sum(Y(:,end)), errKmeans, sum(label2), sum(Y(:,end)));

end

