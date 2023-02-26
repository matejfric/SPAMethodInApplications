function C = compute_C(Gamma,X)
%COMPUTE_C Calculation of the k-means clusters.
n = size(X,1);
K = size(Gamma,1);
C = zeros(n,K);

for k=1:K
    sumGamma = sum(Gamma(k,:), 2);
    if sumGamma ~= 0 
        C(:,k) = X * (Gamma(k,:)') / sumGamma; % notice transposition '
    else
        C(:,k) = zeros(n,1);
    end
end
end

