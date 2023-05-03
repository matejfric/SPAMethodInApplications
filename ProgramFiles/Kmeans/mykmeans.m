function [C, Gamma, sse, it] = mykmeans(X, K, maxIters, eps)
%KMEANS
%   X ....... D x T matrix
%   K ....... number of clusters
%   eps...... tolerance
%   iters ... number of iterations
arguments
    X (:,:) {double}
    K {mustBeNumeric} = 5
    maxIters {mustBeNumeric} = 1e3
    eps {mustBeNumeric} = 1e-6
end

sse = zeros(1,maxIters); sse(1) = Inf; % Sum of squares error
T = size(X,2); % Number of data points in the dataset
[C] = get_kmeans_pp_centroids(X, K); % K-means++ initialization
%[C] = init_random(X', K); C=C';
plot_initialization(X, C);

for it = 2:maxIters+1
        Gamma = zeros(K, T);
        for t = 1:T
            [sse_t,id] = min( sum( (C - X(:,t) ).^2 , 1 ));
            sse(it) = sse(it) + sse_t;
            Gamma(id,t) = 1;
        end
        for k = 1:K
            ids = Gamma(k,:) == 1;
            C(:,k) = mean(X(:,ids),2);
        end
        if norm(sse(it-1) - sse(it)) < eps
            sse = nonzeros(sse(2:end));
            it = it-1;
            break
        end
end
    
end

