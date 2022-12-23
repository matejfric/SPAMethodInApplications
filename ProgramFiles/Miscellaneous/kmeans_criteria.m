%% this function is never used

function [] = kmeans_criteria( X )
%KMEANS_CRITERIA

max_K = 50;

%SSE
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

