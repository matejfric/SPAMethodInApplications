function [C] = kmeans_plus_plus(X, K)
%KMEANS Cluster multivariate data using the k-means++ algorithm.
%   [L,C] = kmeans(X,k) produces a 1-by-size(X,2) vector L with one class
%   label per column in X and a size(X,1)-by-k matrix C containing the
%   centers corresponding to each class.

%   Version: 2013-02-08
%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%   https://www.mathworks.com/matlabcentral/fileexchange/28804-k-means
%
%   References:
%   [1] J. B. MacQueen, "Some Methods for Classification and Analysis of 
%       MultiVariate Observations", in Proc. of the fifth Berkeley
%       Symposium on Mathematical Statistics and Probability, L. M. L. Cam
%       and J. Neyman, eds., vol. 1, UC Press, 1967, pp. 281-297.
%   [2] D. Arthur and S. Vassilvitskii, "k-means++: The Advantages of
%       Careful Seeding", Technical Report 2006-13, Stanford InfoLab, 2006.

    rng(42);
    %X = X(:,1:end-1)';
    
    L = [];
    L1 = 0;

    while length(unique(L)) ~= K

        % The k-means++ initialization.
        C = X(:,1+round(rand*(size(X,2)-1)));
        L = ones(1,size(X,2));
        for i = 2:K
            D = X-C(:,L);
            D = cumsum(sqrt(dot(D,D,1)));
            if D(end) == 0, C(:,i:K) = X(:,ones(1,K-i+1)); return; end
            C(:,i) = X(:,find(rand < D/D(end),1));
            [~,L] = max(bsxfun(@minus,2*real(C'*X),dot(C,C,1).'));
        end
        
    end
end

