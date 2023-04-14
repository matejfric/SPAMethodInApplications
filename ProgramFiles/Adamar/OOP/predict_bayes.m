function [y_pred] = predict_bayes(mdl, X_new)
%ADAMAR_PREDICT Make a prediction based on the Baysian model
%   Lambda.......(M x K) transion matrix
%   X_new........(D x T) matrix of new data

Lambda = mdl.Lambda;
C = mdl.C';
K = size(Lambda,2);
 
% K-means (one step)
dist_from_C = zeros(K, size(X_new,2));
for k=1:K
    dist_from_C(k,:) = sum((X_new - C(:,k)).^2,1);
end
[~,idxY] = min(dist_from_C);
Gamma = zeros(K,length(idxY));
for k = 1:K
   Gamma(k,idxY==k) = 1; 
end

%Pi = round(Lambda*Gamma); % round => binary matrix
Pi = Lambda*Gamma;
y_pred = Pi(1,:);

end

