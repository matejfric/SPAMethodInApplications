function [C] = get_kmeans_pp_centroids(X,K)
%GET_KMEANS_PP_CENTROIDS Get K-means++ initialization of centroids.
    % X ... D x T matrix of data
    % K ... number of clusters
    
    %Returns
    % [C] ... D x K matrix of centroids
    
    % Inspired by
    % https://www.geeksforgeeks.org/ml-k-means-algorithm/
    % https://medium.com/geekculture/implementing-k-means-clustering-with-k-means-initialization-in-python-7ca5a859d63a
    
    %{
    1. The first centroid is selected randomly.
    
    2. Calculate the Euclidean distance between the centroid
       and every other data point in the dataset. The point
       farthest away will become our next centroid.
    
    3. Create clusters around these centroids by associating
       every point with its nearest centroid.
    
    4. The point which has the farthest distance from its centroid
       will be our next centroid.
    
    5. Repeat steps 3 and 4 until K number of centroids are located.
    %}
    
    
    [D, T] = size(X);
    C = zeros(D,K);
    
    % The first centroid is selected randomly
    C(:,1) = X(:, randi([1,T])); 
    %plot_initialization(X, C(:,1));
    
    % Compute remaining K - 1 centroids
    for k = 1:K-1
        dist = zeros(T,1);
        for t = 1:T
            d = Inf;
            
            % Compute distance of data point from each of the previously
            % selected centroids and store the minimum distance
            for kk=1:k
                tmp_dist = sum( (C(:, kk) - X(:,t) ).^2 , 1 );
                d = min(d, tmp_dist);
            end
            dist(t) = d;
        end
        
        %Select data point with maximum distance as our next centroid
        [max_dist, idx_max_dist] = max(dist);
        C(:,k+1) = X(:,idx_max_dist);
        %plot_initialization(X, C(:,1:k+1));
    end
    %plot_initialization(X, C(:,1:k+1));
end

