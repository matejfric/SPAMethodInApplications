function [Gamma, S] = myspa_initial_approximation(X, K)
%INITIAL_APPROXIMATION 
% X     ...D x T
% Gamma ...K x T
% S     ...D x K

[D,T] = size(X);

% Choose random initial centroids

% idx =  randi(T,1,K);
% S = X(:,idx);
[S] = get_kmeans_pp_centroids(X,K);

% Choose random Gamma
idx = randi(K,1,T);
Gamma = onehotencode(categorical(idx),1);

end

