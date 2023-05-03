function [C] = init_random(X, K, seed)
%INIT_RANDOM Random initialization of centroids for k-means 
arguments
    X
    K
    seed = 42
end
rng(seed);

T = size(X,1); % number of points in the dataset
sample_ids = zeros(K,1); % randomly chosen indices for the initial centroids
for k = 1:K
    sample_ids(k) = randi(T);
end
C = X(sample_ids,:); % randomly chosen initial centroids

end

