function Gamma = compute_Gamma_kmeans(C,X)
%COMPUTE_GAMMA Adamar Gamma problem

K = size(C,2);

% K-means (one step)
dist_from_C = zeros(K, size(X,2));
for k=1:K
    dist_from_C(k,:) = sum((X - C(:,k)).^2,1);
end
[~,idxX] = min(dist_from_C);
Gamma = zeros(K,length(idxX));
for k = 1:K
    Gamma(k,idxX==k) = 1;
end

end

