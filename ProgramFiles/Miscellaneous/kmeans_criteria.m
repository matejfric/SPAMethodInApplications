%% this function is never used

function [] = kmeans_criteria(X)
%KMEANS_CRITERIA

max_K = 25;

%SSE
y = zeros(1,max_K);
for k = progress(2:2:max_K)
    [~,~,sumd] = kmeans(X(:,1:4),k, 'MaxIter', 1e3);
    y(k) = sum(sumd);
end
subplot(3,2,[1:2])
plot(1:max_K, y)
ylabel("SSE")
xlabel("Number of Clusters")

%CalinskiHarabasz
evaluation1 = evalclusters(X,"kmeans","CalinskiHarabasz","KList",1:max_K);
fprintf("Optimal K 'CalinskiHarabasz': %d\n", evaluation1.OptimalK);
subplot(3,2,3)
plot(evaluation1)

%SilhouetteCurve
evaluation2 = evalclusters(X,"kmeans","silhouette","KList",1:max_K);
fprintf("Optimal K 'SilhouetteCurve': %d\n", evaluation2.OptimalK);
subplot(3,2,4)
plot(evaluation2)

%DaviesBouldin
evaluation3 = evalclusters(X,"kmeans", "DaviesBouldin","KList",1:max_K);
fprintf("Optimal K 'DaviesBouldin': %d\n", evaluation3.OptimalK);
subplot(3,2,5)
plot(evaluation3)

evaluation4 = evalclusters(X,"kmeans","gap","KList",1:max_K);
subplot(3,2,6)
plot(evaluation4)

%Average number of centroids
number_of_clusters = ceil( (evaluation1.OptimalK +...
    evaluation2.OptimalK + ...
    evaluation3.OptimalK) / 3 );
fprintf("Optimal K: %d\n", number_of_clusters)

