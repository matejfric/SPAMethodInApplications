function [err, errKmeans] = kmeans_svm(X, K)
%KMEANS_SVM Summary of this function goes here
%   X ... matrix of descriptors [n_rows , 1:5]

% Remember scaling for testing dataset
a = min(X(:, 1));
b = max(X(:, 1));
X(:, 1) = (X(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

%% K-means criteria
if false
    %SSE
    max_K = 10;
    y = zeros(1,max_K);
    for k = 1:max_K
        [~,~,sumd] = kmeans(X,k);
        y(k) = sum(sumd);
    end
    figure
    plot(1:max_K, y)
    ylabel("SSE")
    xlabel("Number of Clusters")
    %CalinskiHarabasz
    
    evaluation1 = evalclusters(X,"kmeans","CalinskiHarabasz","KList",1:max_K);
    figure
    fprintf("Optimal K 'CalinskiHarabasz': %d\n", evaluation1.OptimalK);
    plot(evaluation1)
    
    %SilhouetteCurve
    evaluation2 = evalclusters(X,"kmeans","silhouette","KList",1:max_K);
    figure
    fprintf("Optimal K 'SilhouetteCurve': %d\n", evaluation2.OptimalK);
    plot(evaluation2)
    
    %DaviesBouldin
    evaluation3 = evalclusters(X,"kmeans", "DaviesBouldin","KList",1:max_K);
    figure
    fprintf("Optimal K 'DaviesBouldin': %d\n", evaluation3.OptimalK);
    plot(evaluation3)

    %Average number of centroids
    average_K = ceil( (evaluation1.OptimalK +...
        evaluation2.OptimalK + ...
        evaluation3.OptimalK) / 3 );
    fprintf("Optimal K: %d\n", average_K)
end

%%
if (isempty(K))
    K = 10;
end

[idx, C, ~] = kmeans(X(:,1:4),K, 'MaxIter',1000);
cluster_labels = (1:K);
PiX = onehotencode(idx,2,"ClassNames", cluster_labels);

%fprintf("\nPerforming SVM_PiX training...\n")

%[SVMModel_PiX] = fitcsvm(PiX, X(:,5),'Standardize',false,'KernelFunction','polynomial','KernelScale','auto');
[SVMModel_PiX, FitInfo_PiX] = fitclinear(PiX, X(:,5), 'GradientTolerance', 1e-8); % Unsemicolon for results

% fprintf("Performing SVM_PiX cross-validation...\n");
% CVSVMModel_PiX = crossval(SVMModel_PiX);
% classLoss_PiX = kfoldLoss(CVSVMModel_PiX);
% fprintf("ClassLoss = %f\n", classLoss_PiX);
% fprintf("\n");

%fprintf("\nPerforming SVM training...\n");

%SVMModel = fitcsvm(X(:,1:4), X(:,5),'Standardize',false,'KernelFunction','polynomial','PolynomialOrder',2,'KernelScale','auto');
SVMModel = fitcsvm(X(:,1:4), X(:,5),'Standardize',false,'KernelFunction','gaussian','KernelScale','auto');
%[SVMModel, FitInfo] = fitclinear(X(:,1:4), X(:,5), 'GradientTolerance', 1e-8); % Unsemicolon for results
% fprintf("Performing SVM cross-validation...\n");
% CVSVMModel = crossval(SVMModel);
% classLoss = kfoldLoss(CVSVMModel);
% fprintf("ClassLoss = %f\n", classLoss);

%Prediction
ca_predict = cell(1,2);
ca_predict{1,1} = imread('Dataset/Original/68.jpg');
ca_predict{1,2} = imread('Dataset/Annotations/68.png');
Y = roughness_analysis(ca_predict);
Y(:, 1) = (Y(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

[label2,score2] = predict(SVMModel,Y(:,1:4));

%K-means
dist_from_C = zeros(K, size(Y,1));
for k=1:K
    dist_from_C(k,:) = sum((Y(:,1:4)' - C(k,:)').^2,1);
end
[~,idxY] = min(dist_from_C);
GammaY = zeros(K,length(idxY));
for k = 1:K
   GammaY(k,idxY==k) = 1; 
end

% [idx, C, sumd] = kmeans(Y(:,1:4),K, 'MaxIter',1000);
% cluster_labels = (1:K);
% PiX = onehotencode(idx,2,"ClassNames", cluster_labels);
[label1,score1] = predict(SVMModel_PiX, GammaY');

err = norm(abs(label2 - Y(:,5)),1) / size(label1, 1);
errKmeans = norm(abs(label1 - Y(:,5)),1) / size(label1, 1);

fprintf("\nSVM with raw data error:         %.2f (Corroded patches predicted %.2f Vs. true %.2f),\nSVM with Kmeans and Gamma error: %.2f (Corroded patches predicted %.2f Vs. true %.2f)",...
    err, sum(label1), sum(Y(:,5)), errKmeans, sum(label2), sum(Y(:,5)));

end

