function [errY,errX] = kmeans_kl(X, number_of_clusters)
%KMEANS_KL Summary of this function goes here
%   Detailed explanation goes here

% Remember scaling for testing dataset
a = min(X(:, 1));
b = max(X(:, 1));
X(:, 1) = (X(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

%% K-means criteria
if false
    %SSE
    max_K = 50;
    y = zeros(1,max_K);
    for k = 1:max_K
        [~,~,sumd] = kmeans(X(:,1:4),k);
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
    number_of_clusters = ceil( (evaluation1.OptimalK +...
        evaluation2.OptimalK + ...
        evaluation3.OptimalK) / 3 );
    fprintf("Optimal K: %d\n", number_of_clusters)
end

%%

[idx, C] = kmeans(X(:,1:4), number_of_clusters, 'MaxIter',1000);

Gamma = zeros(number_of_clusters,length(idx));
for k = 1:number_of_clusters
   Gamma(k,idx==k) = 1; 
end

Lambda = lambda_solver_jensen(Gamma,[1-X(:,5)'; X(:,5)']);

% Computation of learning error
piX = round(Lambda*Gamma)'; % round => binary matrix
errX = sum(abs(piX(:,2) - X(:,5)))/size(X,1);
disp(['K-means+KL+Jensen learning error = ' num2str(errX)]);

% Prediction
ca_predict = cell(1,2);
ca_predict{1,1} = imread('Dataset/Original/144.jpg');
ca_predict{1,2} = imread('Dataset/Annotations/144.png');
Y = roughness_analysis(ca_predict);
Y(:, 1) = (Y(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

% Kmeans
% dist_from_C = zeros(number_of_clusters, size(Y,1));
% for k=1:number_of_clusters
%     dist_from_C(k,:) = sum((Y(:,1:4)' - C(k,:)').^2,1);
% end
% [~,idxY] = min(dist_from_C);
% GammaY = zeros(number_of_clusters,length(idxY));
% for k = 1:number_of_clusters
%    GammaY(k,idxY==k) = 1; 
% end

dist_from_C = zeros(number_of_clusters, size(Y,1));
for k=1:number_of_clusters
    dist_from_C(k,:) = sum((Y(:,1:4)' - C(k,:)').^2,1);
end
[~,idxY] = min(dist_from_C);
GammaY = zeros(number_of_clusters,length(idxY));
for k = 1:number_of_clusters
   GammaY(k,idxY==k) = 1; 
end

piY = round(Lambda*GammaY)'; % round => binary matrix
errY = sum(abs(piY(:,2) - Y(:,5)))/size(Y,1);
disp(['K-means+KL+Jensen testing error  = ' num2str(errY)]);

end

